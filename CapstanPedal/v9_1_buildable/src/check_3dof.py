"""3-DOF version: the module, its 4-bar leg and the ab/ad drum turn about the ab/ad axis; nothing
that moves may hit the hip frame, motor C, the rail block, its screws or the stand, over the ab/ad
range combined with leg poses. The yoke must also fit the rig like the base mount it replaces."""
import itertools
import math

from build123d import Pos, Rot

import generate as g

ox, oz = g.OUT
mv, fx = g.ab_parts()
legs4 = g.leg4_parts()
hw4 = g.leg_hardware("4bar")
parts = g.build()
ref = g.reference()
A_GROUP = {"Big_Drum_A", "Output_Flange_A", "A_Shaft_Cap"}
B_GROUP = {"Big_Drum_B", "Output_Flange_B"}
rot = lambda p, a, cx, cz: Pos(cx, 0, cz) * Rot(0, -a, 0) * Pos(-cx, 0, -cz) * p      # main XZ, X toward Z
abad = lambda p, phi: Pos(0, g.AB_Y, g.AB_Z) * Rot(phi, 0, 0) * Pos(0, -g.AB_Y, -g.AB_Z) * p   # about X, y toward z


def bb(s):
    b = s.bounding_box()
    return b.min.X, b.min.Y, b.min.Z, b.max.X, b.max.Y, b.max.Z


def touch(a, b):
    return all(a[i] < b[i + 3] and b[i] < a[i + 3] for i in range(3))


_, foot4, _, _ = g.leg4_points(g.L4_Q0["thigh"], g.L4_Q0["shin"])
hip_floor = g.L4_FOOT_R + (oz - foot4[1])                                     # hip height above the floor at rest
fixed = [(n, s, s.solids()) for n, (s, _) in fx.items() if n != "Abad_Motor"]
fixed += [("Abad_Motor", fx["Abad_Motor"][0], [fx["Abad_Motor"][0]])]
fixed += [(n, Pos(0, 0, -(hip_floor - oz)) * s, [Pos(0, 0, -(hip_floor - oz)) * s]) for n, s in g.stand_parts(g.AB_RAIL_X, g.AB_RAIL_Y0 - g.CARRIAGE_Y0).items()]
fixed_boxes = [(n, s, [(so, bb(so)) for so in sols]) for n, s, sols in fixed]


def module_at(ta, tb, qt, qs):
    """Module, leg and ab/ad drum parts (name, solid) at output angles ta, tb (deg) and leg angles."""
    out = []
    for n, (_, pl, _) in parts.items():
        if n.startswith(("Test_Arm", "Base_Mount")):
            continue
        for p in pl:
            p = rot(p, ta, ox, oz) if n in A_GROUP else rot(p, tb, ox, oz) if n in B_GROUP else p
            out.append((n, p))
    for n, s in ref.items():
        if n.startswith("REF_Tensioner_A"):
            s = rot(s, ta, ox, oz)
        elif n.startswith("REF_Tensioner_B"):
            s = rot(s, tb, ox, oz)
        out += [(n, so) for so in s.solids()]
    k0, f0, c0, d0 = g.leg4_points(g.L4_Q0["thigh"], g.L4_Q0["shin"])
    k, f, c, d = g.leg4_points(qt, qs)
    out += [("Leg4_Thigh", rot(legs4["Leg4_Thigh"][1], tb, ox, oz)),
            ("Leg4_Crank", rot(legs4["Leg4_Crank"][1], ta, ox, oz)),
            ("Leg4_Shin", Pos(k[0] - k0[0], 0, k[1] - k0[1]) * rot(legs4["Leg4_Shin"][1], ta, *k0)),
            ("Leg4_Coupler", Pos(c[0] - c0[0], 0, c[1] - c0[1]) * rot(legs4["Leg4_Coupler"][1], tb, *c0))]
    with_shin = lambda p: Pos(k[0] - k0[0], 0, k[1] - k0[1]) * rot(p, ta, *k0)
    for n_ in ("Leg4_Foot", "Leg4_Shin_Plate_Back", "Leg4_Shin_Plate_Front"):
        out.append((n_, with_shin(legs4[n_][1])))
    at = {"knee": (k0, k), "crank": (c0, c), "lever": (d0, d)}
    for n, (cc, _, j, _) in hw4.items():
        out += [(n, with_shin(so) if j == "shin" else Pos(at[j][1][0] - at[j][0][0], 0, at[j][1][1] - at[j][0][1]) * so)
                for so in cc.solids()]
    out += [(n, so) for n, (s, _) in mv.items() for so in s.solids()]
    return out


bad = []
poses = [(259.8, -90.0), (150.0, -90.0), (290.0, -90.0), (259.8, -105.0), (259.8, -75.0)]
for (r, phi_leg), phi in itertools.product(poses, (-20.0, -10.0, 0.0, 10.0, 20.0)):
    qt, qs = g.leg4_ik(r, phi_leg)
    tb, ta = qt - g.L4_Q0["thigh"], qs - g.L4_Q0["shin"]
    hits = []
    for n, s in module_at(ta, tb, qt, qs):
        s = abad(s, phi)
        b = bb(s)
        for m, _, sols in fixed_boxes:
            for so, sb in sols:
                if not touch(b, sb):
                    continue
                x = s & so
                v = x.volume if x is not None else 0.0
                if v > 1.0:
                    hits.append((n, m, round(v, 1)))
    print(f"leg r {r:5.1f} phi {phi_leg:6.1f}  ab/ad {phi:+5.1f}: clashes {sorted(set(hits)) or 'none'}")
    bad += hits

# v6 cradle: the whole module against the spine and the cradle plates at the sector's full +-34 deg
for phi in (-34.0, 34.0):
    qt, qs = g.leg4_ik(259.8, -90.0)
    hits = []
    for n, s in module_at(0.0, 0.0, qt, qs):
        s = abad(s, phi)
        b = bb(s)
        for m, _, sols in fixed_boxes:
            if not m.startswith(("Hip_Spine", "Hip_Front_Plate", "Hip_Inner_Plate", "Hip_Bearing_front")):
                continue
            for so, sb in sols:
                if touch(b, sb):
                    x = s & so
                    if x is not None and x.volume > 1.0:
                        hits.append((n, m, round(x.volume, 1)))
    print(f"module at ab/ad {phi:+5.1f} deg vs spine and cradle plates: clashes {sorted(set(hits)) or 'none'}")
    bad += hits

# the ab/ad capstan itself over the sector's full range: drum vs small drum C, tip bracket and motor C
for phi in (-34.0, -20.0, 0.0, 20.0, 34.0):
    hits = []
    for n in ("Abad_Drum", "Abad_REF_Tensioner_minus_M4x18_screw", "Abad_REF_Tensioner_plus_M4x18_screw",
              "Abad_REF_Tensioner_minus_nut", "Abad_REF_Tensioner_plus_nut"):
        s = abad(mv[n][0], phi)
        for m in ("Abad_Small_Drum", "Hip_Inner_Plate", "Hip_Outer_Plate", "Hip_Posts", "Abad_Motor"):
            x = s & fx[m][0]
            if x is not None and x.volume > 0.5:
                hits.append((n, m, round(x.volume, 1)))
    print(f"ab/ad capstan at {phi:+5.1f} deg: clashes {hits or 'none'}")
    bad += hits

assert not bad, "3-DOF clashes"
print(f"pass: 3-DOF clear over ab/ad +-20 deg with leg poses; module clear of the cradle and ab/ad capstan clear to +-34 deg")
