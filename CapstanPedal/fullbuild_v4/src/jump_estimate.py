"""Jump estimate for the quadruped: mass from the CAD meshes, then a 1-D push along the leg line with
every output torque-limited (28 N*m, as check_strength.py) and speed-limited (8308 KV 90 on 6S through
the 8:1 capstan; the torque falls to zero over the last 15 % of the speed). Gives the take-off speed,
the jump height and air time, and a long jump. Rough: no foot slip, no controller, the steel parts are
taken as solid (bearings are lighter), the motors at ~0.3 kg each.

Run:  python jump_estimate.py <quad_mesh_dir>
"""
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

import make_quad_film as qf

mv = qf.mv
TAU = 28.0                                  # N*m per output
W0 = 2000.0 / 60.0 * 2 * math.pi / 8.0      # no-load output speed, rad/s (KV 90 at 22 V, 8:1)
DENS = {"bulkheads (6061)": 2.7, "carbon plates": 1.6, "printed PETG (60 % effective)": 1.27 * 0.6,
        "aluminium posts and rails (tubes, 45 %)": 2.7 * 0.45, "TPU feet and bumpers": 1.2, "steel (bearings, pins)": 7.8,
        "screws": 7.8, "other (boards, brass)": 3.0}


def mass(meta, tris):
    vol = lambda t: abs(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6) / 1000.0     # cm3
    m = collections.defaultdict(float)
    for k, t in tris.items():
        n, c = k.split("__")[-1], meta[k]["color"]
        if k == "Q_Build_Stand" or c in ("motor", "motor_dark", "magnet"):
            continue
        if n.startswith("Q_") and ("Bulkhead" in n or "End_Frame" in n):
            g = "bulkheads (6061)"
        elif c == "cf":
            g = "carbon plates"
        elif c in ("black", "red"):
            g = "printed PETG (60 % effective)"
        elif c == "dimgray":
            g = "TPU feet and bumpers" if ("Foot" in n or "Bumper" in n) else "aluminium posts and rails (tubes, 45 %)"
        elif c == "blue":
            m["packs (2x 3S 2200)"] = 360.0
            continue
        elif c in ("steel", "silver"):
            g = "steel (bearings, pins)"
        elif c == "screw":
            g = "screws"
        else:
            g = "other (boards, brass)"
        m[g] += vol(t) * DENS[g]
    m["motors (12x 8308, ~0.3 kg)"] = 12 * 300.0
    return {k: v / 1000.0 for k, v in m.items()}


def jac(r, phi):
    qt, qs = mv.leg4_ik(r, phi)
    e = 1e-4
    f = lambda a, b: np.array(mv.leg4_points(a, b)[1], float) / 1000.0
    return np.column_stack([(f(qt + e, qs) - f(qt - e, qs)) / (2 * math.radians(e)),
                            (f(qt, qs + e) - f(qt, qs - e)) / (2 * math.radians(e))])


def push(m, phi=-90.0, r0=230.0, r1=385.0, legs=4):
    """Take-off speed (m/s): four legs extend from r0 to r1 (mm) along the line at phi (deg from +x)."""
    u = np.array([math.cos(math.radians(phi)), math.sin(math.radians(phi))])
    r, v, dt = r0 / 1000.0, 0.0, 1e-4
    while r < r1 / 1000.0:
        J = jac(r * 1000.0, phi)
        w = np.linalg.solve(J, u * v)
        tav = np.array([TAU * min(1.0, max(0.0, (W0 - abs(x)) / (0.15 * W0))) for x in w])
        F = np.min(tav / np.maximum(np.abs(J.T @ u), 1e-9))
        v += (legs * F / m + 9.81 * u[1]) * dt
        r += v * dt
    return v


def main():
    qdir = Path(sys.argv[1]).resolve()
    meta = json.loads((qdir / "quad.json").read_text())["meta"]
    model = qf.build_model(meta)
    _, tris = qf.mesh_data(model, meta)
    ms = mass(meta, tris)
    for k, v in sorted(ms.items(), key=lambda x: -x[1]):
        print(f"  {v:6.2f} kg  {k}")
    tot = sum(ms.values())
    print(f"  {tot:6.2f} kg  total (laser build)")
    F = [TAU / np.max(np.abs(jac(r, -90.0).T @ np.array([0.0, 1.0]))) for r in np.linspace(230.0, 385.0, 50)]
    print(f"one leg, foot under the hip: {min(F):.0f}..{max(F):.0f} N (torque limit only)")
    for m in (tot, tot - 4.0):
        vz, vl = push(m), push(m, phi=-60.0)
        print(f"{m:5.1f} kg: up {vz:.2f} m/s -> {vz * vz / 19.62 * 1000:.0f} mm, {2 * vz / 9.81:.2f} s in the air; "
              f"leaning 30 deg {vl:.2f} m/s -> long jump ~{vl * vl * math.sin(math.radians(120)) / 9.81 * 1000:.0f} mm")


if __name__ == "__main__":
    main()
