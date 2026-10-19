"""Quadruped clash check, as solids (laser build). The four pods (each the v6.2 module without its cradle,
mirrored and with the rear pods' ends swapped as in generate_quad) at ab/ad 0 and with every pod tilted
+-AB_LIMIT together; the body (bulkhead plates, carbon plates, rails, sled, electronics) and the parts of
the ab/ad drives fixed to it (motors C, bearings, posts). Nothing may cut into anything it does not
belong to; the pods may not touch each other.

Run: python check_quad.py
"""
import itertools
import math
import sys

import generate_quad as q
from build123d import Axis, Pos

g = q.g


def bb(s):
    b = s.bounding_box()
    return b.min.X, b.min.Y, b.min.Z, b.max.X, b.max.Y, b.max.Z


def touch(a, b):
    return all(a[i] < b[i + 3] and b[i] < a[i + 3] for i in range(3))


def vol(a, b):
    """Overlap volume, solid by solid (a boolean of whole compounds can report phantom overlaps)."""
    v = 0.0
    for sa in a.solids():
        for sb in b.solids():
            if not touch(bb(sa), bb(sb)):
                continue
            x = sa & sb
            v += x.volume if x is not None else 0.0
    return v


asm = g.assembly("leg_abad_gantry")
abad_of = {}                                                     # which assembly items turn with the ab/ad
mv_, fx_ = g.ab_parts()
pods, fixed = {p: [] for p in q.PODS}, []
for n, (s, grp, col, kind) in asm.items():
    if n.startswith(q.DROP_PREFIX):
        continue
    turns = grp != "world"
    for p in q.PODS:
        placed = q.place(s, p, end=n.startswith(q.END_PREFIX))
        (pods[p] if turns else fixed).append((f"{p}:{n}", placed))
body = [(n, s) for n, (s, c, k) in q.parts(True).items()]
# a pod part may sit in (thread into, run in) these body / fixed parts
OK_PAIRS = (("Abad_Drum", "Abad_Bearings_6808"), ("Hip_Trunnion", "Hip_Bearing_front"), ("Abad_HW_hub_bolts", "Abad_Drum"))


def allowed(a, b):
    return any((x in a and y in b) or (x in b and y in a) for x, y in OK_PAIRS)


def tilt(items, p, deg):
    """Turn a pod's parts about its ab/ad axis (the right pods mirror the sense)."""
    M, t = q.pod_tf(p)
    axis = Axis((0.0, t[1] + M[1, 1] * g.AB_Y, g.AB_Z), (1.0, 0.0, 0.0))
    return [(n, s.rotate(axis, deg * M[1, 1])) for n, s in items]


bad = []
for ab in (0.0, -q.AB_LIMIT, q.AB_LIMIT):
    posed = {p: tilt(pods[p], p, ab) for p in q.PODS}
    hits = []
    for a, b in itertools.combinations(q.PODS, 2):                # pod vs pod
        for (na, sa), (nb, sb) in itertools.product(posed[a], posed[b]):
            if touch(bb(sa), bb(sb)) and vol(sa, sb) > 0.5:
                hits.append((na, nb))
    for p in q.PODS:                                              # pod vs body and the fixed drive parts
        for (na, sa) in posed[p]:
            for (nb, sb) in body + fixed:
                if allowed(na, nb) or not touch(bb(sa), bb(sb)):
                    continue
                v = vol(sa, sb)
                if v > 0.5:
                    hits.append((na, nb, round(v, 1)))
    print(f"ab/ad {ab:+5.1f} deg: {hits or 'clear'}", flush=True)
    bad += hits
hits = []
for (na, sa), (nb, sb) in itertools.combinations(body, 2):        # body parts vs each other
    if touch(bb(sa), bb(sb)) and vol(sa, sb) > 0.5:
        hits.append((na, nb, round(vol(sa, sb), 1)))
for (na, sa) in body:                                             # body vs the fixed drive parts
    for (nb, sb) in fixed:
        if touch(bb(sa), bb(sb)) and vol(sa, sb) > 0.5 and not nb.split(":")[1].startswith(("Abad_Bearing", "Hip_Bearing", "Hip_Posts",
                                                                                           "Hip_HW_post", "Abad_HW_motor",
                                                                                           "Abad_Motor_stator")):
            hits.append((na, nb, round(vol(sa, sb), 1)))
print("body:", hits or "clear", flush=True)
bad += hits
# v3: the body's screws, inserts and nuts against what they do not fasten: the pods (every ab/ad pose), the
# fixed drive parts and the electronics (they may sit in the bulkheads, plates, dock, cassette and lid)
hw = [(f"hw:{n}", s_) for n, s_, *_ in q.hardware()]
others = [(n, s_) for n, s_ in body if n.startswith("REF_")] + fixed
hits = []
for (na, sa) in hw:
    for (nb, sb) in others + [it for ab in (0.0, -q.AB_LIMIT, q.AB_LIMIT) for p in q.PODS for it in tilt(pods[p], p, ab)]:
        if touch(bb(sa), bb(sb)) and vol(sa, sb) > 0.2:
            hits.append((na, nb, round(vol(sa, sb), 1)))
print("body hardware:", hits or "clear", flush=True)
bad += hits
print("PASS" if not bad else f"FAIL: {len(bad)}")
sys.exit(0 if not bad else 1)
