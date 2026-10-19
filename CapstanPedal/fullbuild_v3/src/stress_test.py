"""Stress test of the quadruped body (fullbuild_v3): linear-elastic FEA of the three bulkheads that carry the
pods (centre bulkhead, inner bulkhead, end frame; the rear set mirrors the front), with the same solver as
the leg module's fea_check.py (gmsh 2nd-order tetrahedra, scikit-fem, AMG-preconditioned CG).

Load path. A foot force F (vertical, up) goes into its pod, which hangs on three bearings in the body: the
ab/ad hub's two 6808s (centre and inner bulkheads) and the trunnion's 6806 (end frame). The pod is taken as a
stiff beam on those three supports with F under the hip; the reactions come from the continuous-beam
solution. The ab/ad drive holds the leg's lean (up to its 31.5 N*m limit) through motor C's rope, whose pull
(31.5 N*m over the 70 mm drum) loads motor C's stator mount (centre bulkhead) and its tip bearing (inner
bulkhead), half each. Each bulkhead carries its left and right pods together.

Load cases (per foot):
  push / landing  F = 585 N (the most a leg can push: both outputs at 31.5 N*m, leg extended)  target SF 2
  impact x2       F = 1170 N on both pods                                                   target SF 1 (no yield)
  one leg x2      F = 1170 N on the left pod only, the right unloaded (twist)               target SF 1
Supports: each bulkhead is held where the top and bottom plates bear on it (its top and bottom edges and tabs).
Materials, from the same solution (force loads on a held part: the stress does not depend on E; Poisson's
ratio 0.38 is used): printed PETG (45 MPa along the layers, 25 across; prints flat, layers across X) and 6061-T6
(yield 276 MPa) for the laser build's plate pairs, taken as bonded.

Run: python stress_test.py           (writes ../stress_test/results.md and results.json)
"""
import json
import math
import os
import sys

import numpy as np

import generate_quad as q

g = q.g
sys.path.insert(0, os.path.dirname(g.__file__))
import fea_check as fc                                                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "stress_test")
F_PUSH = 520.0 * 31.5 / 28.0                 # v3: 9:1 drives, 31.5 N*m per output (v2: 28 -> 520 N)
TAU_AB = 31.5
AL_YIELD = 276.0
X = (1.0, 0.0, 0.0)


def beam_reactions(xs, xf, F):
    """Reactions of a stiff-ish beam on simple supports at xs (3) under F at xf: the continuous-beam solution
    (the middle support as the redundant, deflections from the simply supported span between the outer two)."""
    a, b, c = sorted(xs)
    L = c - a

    def defl(x, d, P):                                                    # simply supported span a..c, load P at d
        u, v = d - a, c - d
        x = x - a
        return P * v * x * (L * L - v * v - x * x) / (6 * L) if x <= u else P * u * (L - x) * (2 * L * x - x * x - u * u) / (6 * L)
    Rb = defl(b, xf, F) / defl(b, b, 1.0)
    Ra = (F * (c - xf) - Rb * (c - b)) / L
    Rc = F - Ra - Rb
    return {a: Ra, b: Rb, c: Rc}


def geometry():
    """Front set: x of each support, the pods' ab/ad axes (y, z), motor C's axis (y, z) on the left."""
    xc, xi, xe = (sum(q.bulkhead_x(k, True)) / 2 for k in ("outer", "inner", "front"))
    M, t = q.pod_tf("LF")
    y_ax = float(t[1] + M[1, 1] * g.AB_Y)
    asm = g.ab_parts()[1]
    sd = q.place(asm["Abad_Small_Drum"][0], "LF", end=True).bounding_box()
    mc = (round(float(sd.min.Y + sd.max.Y) / 2, 2), round(float(sd.min.Z + sd.max.Z) / 2, 2))
    hip_x = q.DX + g.OUT[0]
    return dict(x={"outer": xc, "inner": xi, "front": xe}, y_ax=y_ax, z_ax=g.AB_Z, motorC=mc, hip_x=hip_x)


def loads_for(p, kind, G, cases):
    """FEA load vectors on bulkhead `kind` for each case: [(tag, f)]."""
    xa, xb = q.bulkhead_x(kind, True)
    span = (xa, xb)
    xs = [G["x"]["outer"], G["x"]["inner"], G["x"]["front"]]
    r_seat = {"outer": (g.B40[1] + g.PRESS) / 2, "inner": (g.B40[1] + g.PRESS) / 2, "front": (g.B30[1] + g.PRESS) / 2}[kind]
    T = TAU_AB / (g.BIG_PITCH_R / 1000.0)                                  # motor C's rope pull, N
    out = []
    for tag, (FL, FR) in cases:
        f = np.zeros(p.K.shape[0])
        for sy, F in ((1, FL), (-1, FR)):
            if F == 0:
                continue
            R = beam_reactions(xs, G["hip_x"], F)[G["x"][kind]]
            c = (0.0, sy * G["y_ax"], G["z_ax"])
            sel = p.bore(c, X, r_seat, span, tol=0.6)
            if len(sel) == 0:
                raise RuntimeError(f"{kind}: no bearing seat found at {c} r {r_seat}")
            f += p.bearing_load(sel, c, X, (0.0, 0.0, R))
            if kind in ("outer", "inner"):                                 # motor C: half the rope pull each end
                my, mz = G["motorC"]
                cm = (0.0, sy * my, mz)
                d = np.array([0.0, sy * G["y_ax"] - cm[1], G["z_ax"] - cm[2]])
                d = d / np.linalg.norm(d)
                if kind == "outer":
                    sel = p.face(X, xb, (xb, cm[1], cm[2]), 34.0, tol=0.3)       # stator seat on the motor face
                    f += p.spread_load(sel, 0.5 * T * d)
                else:
                    sel = p.bore(cm, X, (g.B15[1] + g.PRESS) / 2, span, tol=0.6)  # tip bearing
                    f += p.bearing_load(sel, cm, X, 0.5 * T * d)
        out.append((tag, f))
    return out


def supports(p):
    zt, zb = q.Z_TOP, q.Z_BOT
    top = p.bf[p.fc[2] > zt - 0.3]
    bot = p.bf[p.fc[2] < zb + 0.3]
    return [top, bot]


def main():
    os.makedirs(OUT, exist_ok=True)
    G = geometry()
    cases = [(f"push / landing {F_PUSH:.0f} N", (F_PUSH, F_PUSH)), (f"impact x2 {2 * F_PUSH:.0f} N", (2 * F_PUSH, 2 * F_PUSH)),
             (f"one leg x2 {2 * F_PUSH:.0f} N", (2 * F_PUSH, 0.0))]
    target = {cases[0][0]: 2.0, cases[1][0]: 1.0, cases[2][0]: 1.0}
    pl = q.cradle_plates()
    rows = []
    for kind, name in (("outer", "Centre bulkhead"), ("inner", "Inner bulkhead"), ("front", "End frame")):
        solid = q.bulkhead(kind, True, pl)
        p = fc.Part(name, solid, 3.0, X)
        fixed = supports(p)
        lds = loads_for(p, kind, G, cases)
        sols, _ = p.solve(fixed, lds)
        print(f"[{name}]", flush=True)
        for tag, u in sols:
            vm, sz, vol, near, cen = p.stresses(u, fixed)
            vm_p, vm_max = fc.peak(vm, vol, near)
            sz_p, _ = fc.peak(sz, vol, near)
            sf_petg = min(fc.PETG_XY / vm_p, fc.PETG_Z / max(sz_p, 1e-9))
            sf_al = AL_YIELD / vm_p
            k = int(np.argmax(np.where(near, 0, vm)))
            umax = float(np.abs(u[p.ib.nodal_dofs]).max())
            row = dict(part=name, case=tag, vm=round(float(vm_p), 1), across=round(float(sz_p), 1), sf_petg=round(float(sf_petg), 2),
                       sf_6061=round(float(sf_al), 1), disp_petg=round(umax, 2), disp_6061=round(umax * fc.E / 68900.0, 3),
                       hot=[round(float(v)) for v in cen[:, k]], target=target[tag])
            rows.append(row)
            print(f"  {tag:22s} von Mises {vm_p:6.1f} MPa  across layers {sz_p:5.1f}  SF PETG {sf_petg:4.1f}  SF 6061 {sf_al:5.1f}  "
                  f"disp PETG {umax:5.2f} mm / 6061 {row['disp_6061']:.3f} mm  hottest at {row['hot']}", flush=True)
    R = beam_reactions([G["x"]["outer"], G["x"]["inner"], G["x"]["front"]], G["hip_x"], F_PUSH)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(dict(geometry={k: (v if not isinstance(v, tuple) else list(v)) for k, v in G.items()},
                       reactions_push={f"{k:.2f}": v for k, v in R.items()}, rows=rows), f, indent=1)
    lines = ["# Body stress test (fullbuild_v3): results", "",
             "Linear-elastic FEA (`src/stress_test.py`; gmsh P2 tetrahedra, scikit-fem). Peak von Mises with the hottest 0.1 % of",
             "the volume and the elements within 2.5 mm of the held faces left out. SF: PETG against 45 MPa along / 25 MPa across",
             "the layers; 6061-T6 against 276 MPa yield. Displacements: PETG (E 2000 MPa) and 6061 (E 68.9 GPa).", "",
             f"Pod bearing reactions per foot at {F_PUSH:.0f} N (continuous beam, foot under the hip at x {G['hip_x']:.1f}): " +
             ", ".join(f"x {k:.1f}: {v:.0f} N" for k, v in R.items()) + f"; motor C rope pull {TAU_AB / 0.07:.0f} N (half per end).", "",
             "| Part | Case | von Mises MPa | across layers MPa | SF printed (PETG) | SF laser (6061) | max disp PETG / 6061 mm | target | pass (printed / laser) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ok = lambda sf: "✓" if sf >= r["target"] else "✗"
        lines.append(f"| {r['part']} | {r['case']} | {r['vm']} | {r['across']} | {r['sf_petg']} | {r['sf_6061']} | {r['disp_petg']} / {r['disp_6061']} | "
                     f"{r['target']} | {ok(r['sf_petg'])} / {ok(r['sf_6061'])} |")
    with open(os.path.join(OUT, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("written", os.path.join(OUT, "results.md"))


if __name__ == "__main__":
    main()
