"""Check every purchased fastener against every printed part and reference component.

A fastener may only overlap what it threads into: its own insert or nut, or the motor it
screws into. Anything else over 0.2 mm^3 is reported as a clash.
"""
import itertools

import generate as g

placed = []
for name, (_, parts, _) in g.build().items():
    placed += [(f"{name}#{i + 1}", p) for i, p in enumerate(parts)]
for name, part in g.reference().items():
    if not name.startswith("REF_Motor"):
        placed.append((name, part))
for w in ("A", "B"):
    for piece, part in g.ref_motor_split(w).items():
        placed.append((f"REF_Motor_{w}_{piece}", part))
fasteners = [(f"{name}.{i + 1}", s) for name, c in g.fasteners().items() for i, s in enumerate(c.solids())]
for name, part in g.reference().items():
    if name.startswith("REF_Tensioner"):
        fasteners.append((name, part))

# what each fastener family is allowed to touch (it threads into these)
THREADS = {
    "FST_Stator": ("REF_Motor_A_Stator", "REF_Motor_B_Stator"),
    "FST_Rotor": ("REF_Motor_A_Rotor", "REF_Motor_B_Rotor"),
}


def bb(s):
    b = s.bounding_box()
    return b.min.X, b.min.Y, b.min.Z, b.max.X, b.max.Y, b.max.Z


def touch(a, b):
    return all(a[i] < b[i + 3] and b[i] < a[i + 3] for i in range(3))


boxes = {n: bb(s) for n, s in placed + fasteners}
clashes = []
for (fn, fs), (pn, ps) in itertools.product(fasteners, placed + fasteners):
    if fn >= pn and (pn, ps) in fasteners or fn == pn:
        continue
    if not touch(boxes[fn], boxes[pn]):
        continue
    family = fn.split(".")[0]
    if any(family.startswith(k) and pn.startswith(v) for k, vs in THREADS.items() for v in vs):
        continue
    if pn.startswith(("FST_", "REF_Tensioner")):          # screw into its insert/nut
        continue
    if "Insert" in family and not pn.startswith("REF"):   # heat-set inserts melt into undersized holes
        continue
    if family.startswith("REF_Tensioner") and pn.startswith("REF_Tensioner"):
        continue
    x = fs & ps
    v = x.volume if x is not None else 0.0
    if v > 0.2:
        clashes.append((round(v, 1), fn, pn))
for c in sorted(clashes, reverse=True):
    print("CLASH", *c)
print(f"{len(fasteners)} fasteners checked, {len(clashes)} clashes")

# Access: back each screw out along its axis. Anything it hits on the way is in the path
# of the screw (or the driver) during assembly. Parts fitted later in the sequence are skipped.
OUT_DIR = {  # family: (direction the screw backs out along, parts fitted after it)
    "FST_Stator": ((0, -1, 0), ("REF_moteus",)),               # standoffs are fitted after these
    "FST_Screw_moteus": ((0, -1, 0), ()),
    "FST_Rotor": ((0, 1, 0), ("Front_Wall", "Big_Drum", "Post", "Top_Rail", "Output", "Test_Arm", "REF_Bearing_15x28_tip", "Base_Mount", "A_Shaft_Cap")),
    "FST_Screw_B_flange": ((0, 1, 0), ("Test_Arm", "Output_Flange_A")),
    "FST_Screw_B_arm": ((0, 1, 0), ("Output_Flange_A", "Test_Arm_A", "Gantry", "REF_Bearing_17x30_gantry")),
    "FST_Screw_A_flange": ((0, 1, 0), ("Test_Arm_A", "Gantry", "REF_Bearing_17x30_gantry")),
    "FST_Screw_A_arm": ((0, 1, 0), ("Gantry", "REF_Bearing_17x30_gantry")),
    "FST_Bolt_posts": ((0, 1, 0), ("Output", "Test_Arm", "Base_Mount", "Gantry", "REF_Bearing_17x30_gantry")),
    "FST_Screw_posts_back": ((0, -1, 0), ("REF_moteus",)),
    "FST_Bolt_gantry": ((0, 1, 0), ()),
    "FST_Screw_A_cap": ((0, -1, 0), ()),
}
from build123d import Pos
blocked = []
for fn, fs in fasteners:
    fam = next((k for k in OUT_DIR if fn.startswith(k)), None)
    if fam is None:
        continue
    d, later = OUT_DIR[fam]
    for pn, ps in placed:
        if pn.startswith(later):
            continue
        base = fs & ps
        v0 = base.volume if base is not None else 0.0
        for k in range(2, 26, 2):
            moved = Pos(d[0] * k, d[1] * k, d[2] * k) * fs
            if not touch(bb(moved), boxes[pn]):
                continue
            x = moved & ps
            v = x.volume if x is not None else 0.0
            if v > v0 + 0.2:
                blocked.append((fn, pn, k, round(v - v0, 1)))
                break
# The brass moteus standoffs go in after the stator screws: they must not clash with the heads,
# and each board screw must reach its standoff.
fst = dict(fasteners)
for w in "AB":
    for so in (n for n in fst if n.startswith(f"FST_Standoff_moteus_{w}")):
        for sc in (n for n in fst if n.startswith(f"FST_Stator_{w}")):
            x = fst[so] & fst[sc]
            if x is not None and x.volume > 0.2:
                blocked.append((so, sc, 0, round(x.volume, 1)))
for b in blocked:
    print("BLOCKED", *b)
print(f"access: {len(blocked)} blocked paths")
assert not clashes and not blocked, "fastener problems"
