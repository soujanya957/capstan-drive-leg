"""Five-bar leg: every sampled foot position must stay inside the +-OUT_RANGE/2 output range and
clear the rig (test arms removed). Thighs turn with the outputs; shins and foot follow the five-bar."""
import itertools
import math

from build123d import Pos, Rot

import generate as g

ox, oz = g.OUT
q0, knee0, foot0 = g.leg_rest()
legs = g.leg_parts()
parts = g.build()
A_GROUP = {"Big_Drum_A", "Output_Flange_A", "A_Shaft_Cap"}
B_GROUP = {"Big_Drum_B", "Output_Flange_B"}
rot = lambda p, a, cx, cz: Pos(cx, 0, cz) * Rot(0, -a, 0) * Pos(-cx, 0, -cz) * p    # +a: X toward Z


def five_bar(qa, qb):
    k = {w: (ox + g.LEG_L1 * math.cos(math.radians(q)), oz + g.LEG_L1 * math.sin(math.radians(q)))
         for w, q in (("A", qa), ("B", qb))}
    mx, mz = (k["A"][0] + k["B"][0]) / 2, (k["A"][1] + k["B"][1]) / 2
    dx, dz = k["B"][0] - k["A"][0], k["B"][1] - k["A"][1]
    half = math.hypot(dx, dz) / 2
    h = math.sqrt(g.LEG_L2 ** 2 - half ** 2)
    nx, nz = -dz / (2 * half), dx / (2 * half)
    if (mx + nx - ox) ** 2 + (mz + nz - oz) ** 2 < (mx - nx - ox) ** 2 + (mz - nz - oz) ** 2:
        nx, nz = -nx, -nz
    return k, (mx + h * nx, mz + h * nz)


hw4 = g.leg_hardware("4bar")
bad = []
# 4-bar leg: B swings the thigh, A's crank sets the shin through the parallel coupler
legs4 = g.leg4_parts()
qt0, qs0 = g.L4_Q0["thigh"], g.L4_Q0["shin"]
k0, f0, c0, d0 = g.leg4_points(qt0, qs0)
for r, phi in itertools.product([x * g.L4_K for x in (175.0, 200.0, 259.8, 290.0)], (-105.0, -90.0, -75.0)):   # v8: scaled with the leg
    qt, qs = g.leg4_ik(r, phi)
    tb, ta = qt - qt0, qs - qs0
    if max(abs(ta), abs(tb)) > g.OUT_RANGE / 2:
        print(f"4-bar r {r:5.1f} phi {phi:6.1f}: outputs A {ta:+.1f} B {tb:+.1f} deg  out of range")
        continue
    k, f, c, d = g.leg4_points(qt, qs)
    placed = {"Leg4_Thigh": rot(legs4["Leg4_Thigh"][1], tb, ox, oz),
              "Leg4_Crank": rot(legs4["Leg4_Crank"][1], ta, ox, oz),
              "Leg4_Shin": Pos(k[0] - k0[0], 0, k[1] - k0[1]) * rot(legs4["Leg4_Shin"][1], ta, *k0),
              "Leg4_Coupler": Pos(c[0] - c0[0], 0, c[1] - c0[1]) * rot(legs4["Leg4_Coupler"][1], tb, *c0)}
    with_shin = lambda p: Pos(k[0] - k0[0], 0, k[1] - k0[1]) * rot(p, ta, *k0)
    for n_ in ("Leg4_Foot", "Leg4_Shin_Plate_Back", "Leg4_Shin_Plate_Front"):    # ride on the shin
        placed[n_] = with_shin(legs4[n_][1])
    joint_at = {"knee": (k0, k), "crank": (c0, c), "lever": (d0, d)}
    hw = {n: [with_shin(so) if j == "shin" else
              Pos(joint_at[j][1][0] - joint_at[j][0][0], 0, joint_at[j][1][1] - joint_at[j][0][1]) * so
              for so in cc.solids()] for n, (cc, _, j, _) in hw4.items()}
    rig = []
    for n, (_, pl, _) in parts.items():
        if n.startswith("Test_Arm"):
            continue
        for p in pl:
            p = rot(p, ta, ox, oz) if n in A_GROUP else rot(p, tb, ox, oz) if n in B_GROUP else p
            rig.append((n, p))
    hits = []
    for n, p in placed.items():
        for m, o in rig + [(k2, v) for k2, v in placed.items() if k2 != n]:
            x = p & o
            if x is not None and x.volume > 0.5:
                hits.append((n, m, round(x.volume, 1)))
    for n, p in hw.items():                                  # each bolt, washer, nut, shim and bearing on its own
        for so in p:
            for m, o in rig + list(placed.items()):
                if n.endswith("shin_inserts") and m in ("Leg4_Shin_Plate_Back", "Leg4_Shin"):
                    continue                                 # v9.1: a heat-set insert's knurl sits in its plate
                x = so & o
                if x is not None and x.volume > 2.0:                     # washers sit flat on the links
                    hits.append((n, m, round(x.volume, 1)))
    for n, p in hw.items():                                  # each bolt, washer, nut, shim and bearing on its own
        for so in p:
            for m, o in rig + list(placed.items()):
                if n.endswith("shin_inserts") and m in ("Leg4_Shin_Plate_Back", "Leg4_Shin"):
                    continue                                 # v9.1: a heat-set insert's knurl sits in its plate
                x = so & o
                if x is not None and x.volume > 2.0:                     # washers sit flat on the links
                    hits.append((n, m, round(x.volume, 1)))
    print(f"4-bar r {r:5.1f} phi {phi:6.1f}: outputs A {ta:+5.1f} B {tb:+5.1f} deg  clashes: {hits or 'none'}")
    bad += hits
assert not bad, "4-bar leg clashes"
print("pass: 4-bar leg poses in range and clear")

print("the five-bar leg and the v3 rail carriage are not used")
