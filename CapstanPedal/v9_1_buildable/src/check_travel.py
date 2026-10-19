"""v8: the 4-bar leg over the outputs' whole travel. Both outputs are swept over +-OUT_RANGE/2 on a grid;
at each pose the leg's links (thigh, crank, shin and its plates, coupler, foot) are checked against the
module (drums turned with their outputs) and against each other. Poses where the 4-bar folds through
its parallelogram's dead centre (coupler and thigh within 10 deg of collinear with the crank) are
reported as such, not as clashes. Prints a map of the clear poses.

Run: CAPSTAN_LEG_L=200 python check_travel.py [step_deg]
"""
import itertools
import math
import sys

from build123d import Pos, Rot

import generate as g

STEP = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
ox, oz = g.OUT
qt0, qs0 = g.L4_Q0["thigh"], g.L4_Q0["shin"]
k0, f0, c0, d0 = g.leg4_points(qt0, qs0)
legs4 = g.leg4_parts()
parts = g.build()
A_GROUP = {"Big_Drum_A", "Output_Flange_A", "A_Shaft_Cap"}
B_GROUP = {"Big_Drum_B", "Output_Flange_B"}
rot = lambda p, a, cx, cz: Pos(cx, 0, cz) * Rot(0, -a, 0) * Pos(-cx, 0, -cz) * p    # +a: X toward Z
R = g.OUT_RANGE / 2
KNEE = (60.0, 60.0)                   # knee range (thigh minus shin output, deg from rest): past +60 the thigh meets the shin plate, at -75 the 4-bar reaches dead centre


def bb(s):
    b = s.bounding_box()
    return b.min.X, b.min.Y, b.min.Z, b.max.X, b.max.Y, b.max.Z


def touch(a, b):
    return all(a[i] < b[i + 3] and b[i] < a[i + 3] for i in range(3))


def pose(ta, tb):
    """Placed leg links and the rig, outputs A (crank / shin) at ta and B (thigh) at tb from rest."""
    qt, qs = qt0 + tb, qs0 + ta
    k, f, c, d = g.leg4_points(qt, qs)
    with_shin = lambda p: Pos(k[0] - k0[0], 0, k[1] - k0[1]) * rot(p, ta, *k0)
    placed = {"Leg4_Thigh": rot(legs4["Leg4_Thigh"][1], tb, ox, oz),
              "Leg4_Crank": rot(legs4["Leg4_Crank"][1], ta, ox, oz),
              "Leg4_Coupler": Pos(c[0] - c0[0], 0, c[1] - c0[1]) * rot(legs4["Leg4_Coupler"][1], tb, *c0)}
    for n in ("Leg4_Shin", "Leg4_Foot", "Leg4_Shin_Plate_Back", "Leg4_Shin_Plate_Front"):
        placed[n] = with_shin(legs4[n][1])
    rig = []
    for n, (_, pl, _) in parts.items():
        if n.startswith(("Test_Arm", "Leg")):
            continue
        for p in pl:
            p = rot(p, ta, ox, oz) if n in A_GROUP else rot(p, tb, ox, oz) if n in B_GROUP else p
            rig.append((n, p))
    # dead centre: crank direction vs the thigh (the coupler stays parallel to the thigh)
    crank = qs + 180.0 + g.L4_DELTA
    dead = min(abs((crank - qt + 180.0) % 360.0 - 180.0), abs((crank - qt) % 360.0 - 180.0)) < 10.0
    return placed, rig, dead, (k, f)


def clashes(placed, rig):
    hits = []
    links = list(placed.items())
    for n, p in links:
        pb = bb(p)
        for m, o in rig:
            if touch(pb, bb(o)):
                x = p & o
                if x is not None and x.volume > 0.5:
                    hits.append((n, m, round(x.volume, 1)))
    for (n, p), (m, o) in itertools.combinations(links, 2):
        if {n, m} <= {"Leg4_Shin", "Leg4_Foot", "Leg4_Shin_Plate_Back", "Leg4_Shin_Plate_Front"}:
            continue                                               # one rigid body
        if touch(bb(p), bb(o)):
            x = p & o
            if x is not None and x.volume > 0.5:
                hits.append((n, m, round(x.volume, 1)))
    return hits


def main():
    angles = [a for a in [-R + i * STEP for i in range(int(2 * R / STEP) + 1)]]
    grid = {}
    for ta in angles:
        for tb in angles:
            placed, rig, dead, (k, f) = pose(ta, tb)
            if not -KNEE[0] <= tb - ta <= KNEE[1]:
                grid[(ta, tb)] = "-"
                print(f"A {ta:+6.1f} B {tb:+6.1f}: outside the knee range", flush=True)
                continue
            if dead:
                grid[(ta, tb)] = "D"
                print(f"A {ta:+6.1f} B {tb:+6.1f}: dead centre", flush=True)
                continue
            h = clashes(placed, rig)
            grid[(ta, tb)] = "." if not h else "X"
            print(f"A {ta:+6.1f} B {tb:+6.1f}: foot ({f[0]:6.1f}, {f[1]:6.1f})  {h or 'clear'}", flush=True)
    print("\nmap (rows: output A / crank+shin, columns: output B / thigh; . clear, X clash, D dead centre, - outside the knee range)")
    print("        " + " ".join(f"{tb:+5.0f}" for tb in angles))
    for ta in angles:
        print(f"{ta:+6.0f}  " + " ".join(f"{grid[(ta, tb)]:>5s}" for tb in angles))


if __name__ == "__main__":
    main()
