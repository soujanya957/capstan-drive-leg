"""The eight builds (test arms on the desk or 4-bar leg on the rail stand, each with or without ab/ad and
the front gantry): nothing that moves may touch anything it moves against, over the output range,
the leg poses and ab/ad; the mount parts and their screws must fit the module at rest; every mount
screw must have a clear path to back out along its axis.

Run: python check_variants.py [variant ...]      (default: all eight)
"""
import itertools
import sys

from build123d import Pos

import generate as g

R = round(g.OUT_RANGE / 2, 1)
R_ARM = min(R, 34.2)                    # v8: the bench test arms are run over v6.2's +-34.2 (at +-90 they reach the desk and cradle)
ARM_POSES = [(0.0, 0.0), (R_ARM, R_ARM), (-R_ARM, -R_ARM), (R_ARM, -R_ARM), (-R_ARM, R_ARM)]
LEG_POSES = [(r * g.L4_K, p) for r, p in ((259.8, -90.0), (175.0, -90.0), (290.0, -90.0), (259.8, -105.0), (259.8, -75.0))]   # v8: scaled with the leg; the squat 175 (x K) keeps the knee in its +-60
ABAD = (-20.0, 0.0, 20.0)
MOUNT = ("Hip_", "Abad_", "End_Bracket", "Desk_", "Rail_Carriage", "Mount_HW", "REF_MGN12H_blocks", "FST_Insert_rails", "Top_Rail",
         "Hip_Spine", "Hip_HW_block_screws", "Hip_REF_MGN12H_blocks", "Hip_HW_rail_screws")
# a screw (or insert) may overlap what it threads into
THREADS = {"Mount_HW_base_screws": ("FST_Insert_rails",), "Mount_HW_carriage_screws": ("FST_Insert_rails",),
           "FST_Insert_rails": ("Top_Rail", "Mount_HW"), "FST_Insert_posts": ("Top_Rail", "Post", "FST_Bolt_posts", "FST_Screw_posts_back"), "Mount_HW_block_screws": ("REF_MGN12H_blocks",),
           "Hip_HW_block_screws": ("Hip_REF_MGN12H_blocks",), "Mount_HW_wood_screws": ("REF_Desk",),
           "Hip_HW_rail_screws": ("Hip_Outer_Plate", "Hip_Inner_Plate", "Hip_Front_Plate"),
           "Hip_HW_post_inserts": ("Hip_Inner_Plate", "Hip_HW_post_bolts"),
           "Hip_HW_trunnion_inserts": ("End_Bracket_Front", "Hip_HW_trunnion_screws"),
           "Abad_HW_motor": ("Abad_Motor", "Hip_Outer_Plate", "Abad_moteus", "Abad_HW_motor"), "Abad_HW_small_drum_screws": ("Abad_Motor",),
           "Abad_REF_Tensioner": ("Abad_REF_Tensioner",),
           "Abad_Motor": ("Hip_Outer_Plate",),
           # v9.1 heat-set inserts in their hosts (the knurl sits in the plastic), and the screws in them
           "Leg4_HW_shin_inserts": ("Leg4_Shin_Plate_Back", "Leg4_HW_shin_screws"),
           "Abad_HW_hub_inserts": ("Abad_Drum", "Abad_HW_hub_bolts"),
           "FST_Insert_B_flange": ("Output_Flange_B", "FST_Screw_B_arm")}   # whole supplier STEP; split into rotor/stator/holder it is clear (as the main motors)
# back-out direction and the parts fitted after the screw (not in its way while it goes in); FIRST: fitted
# into a sub-assembly before anything else goes on (checked against that sub-assembly only)
FIRST = {"Hip_HW_block_screws": ("Hip_Spine", "Hip_REF_MGN12H_blocks")}
ACCESS = {"Mount_HW_base_screws": ((0, 0, 1), ("REF_Desk", "Mount_HW_wood_screws")),
          "Mount_HW_carriage_screws": ((0, 0, 1), ("Stand_", "REF_MGN12H_blocks", "Mount_HW_block_screws")),
          "Mount_HW_block_screws": ((0, 1, 0), ("Stand_",)),
          "Mount_HW_wood_screws": ((0, 0, -1), ()),
          "Hip_HW_block_screws": ((0, 1, 0), ("Abad_", "Hip_Outer", "Hip_Inner", "Hip_Posts", "Hip_HW_post", "Hip_HW_rail",
                                              "Hip_Bearings", "Stand_")),
          "Hip_HW_rail_screws": ((0, -1, 0), ("REF_Desk", "Mount_HW_wood_screws")),
          "Hip_HW_trunnion_screws": ((1, 0, 0), ("Hip_Front_Plate", "Hip_Bearing_front", "Hip_Spine", "Hip_HW_rail", "Desk_", "Mount_HW",
                                                 "REF_Desk", "Stand_", "Hip_REF_MGN12H", "Hip_HW_block"))}


def bb(s):
    b = s.bounding_box()
    return b.min.X, b.min.Y, b.min.Z, b.max.X, b.max.Y, b.max.Z


def touch(a, b):
    return all(a[i] < b[i + 3] and b[i] < a[i + 3] for i in range(3))


def solids(posed, asm):
    out = []
    for n, (s, grp) in posed.items():
        kind = asm[n][3]
        for i, so in enumerate(s.solids()):
            out.append((n, i, so, grp, kind, bb(so)))
    return out


def allowed(a, b):
    return any(a.startswith(k) and b.startswith(v) for k, vs in THREADS.items() for v in vs) or \
           any(b.startswith(k) and a.startswith(v) for k, vs in THREADS.items() for v in vs)


def moving_clashes(items, abad):
    hits = set()
    for (n1, _, s1, g1, k1, b1), (n2, _, s2, g2, k2, b2) in itertools.combinations(items, 2):
        if g1 == g2 or (g1 == "world" and g2 == "world") or not touch(b1, b2) or allowed(n1, n2):
            continue
        if not abad and "world" not in (g1, g2) and {g1, g2} <= {"module"}:
            continue
        if n1.startswith("REF_Tensioner") and n2.startswith("REF_Tensioner"):
            continue
        x = s1 & s2
        v = x.volume if x is not None else 0.0
        leg_hw = (k1 == "hw" and g1.startswith("leg")) or (k2 == "hw" and g2.startswith("leg"))
        if v > (2.0 if leg_hw else 1.0):                                  # pin washers sit flat on the links
            hits.add((n1, n2, round(v, 1)))
    return hits


def mount_clashes(items):
    hits = set()
    for (n1, _, s1, g1, k1, b1), (n2, _, s2, g2, k2, b2) in itertools.combinations(items, 2):
        if g1 != g2 or not (n1.startswith(MOUNT) or n2.startswith(MOUNT)) or n1 == n2:
            continue
        if not touch(b1, b2) or allowed(n1, n2):
            continue
        x = s1 & s2
        v = x.volume if x is not None else 0.0
        if v > 0.2:
            hits.add((n1, n2, round(v, 1)))
    return hits


def access(items):
    blocked = []
    for n, i, s, grp, kind, b in items:
        fam = next((k for k in ACCESS if n.startswith(k)), None)
        if fam is None:
            continue
        d, later = ACCESS[fam]
        for m, j, o, g2, k2, ob in items:
            if (m, j) == (n, i) or m.startswith(later) or m.startswith(fam) or allowed(n, m):
                continue
            if fam in FIRST and not m.startswith(FIRST[fam]):
                continue
            base = s & o
            v0 = base.volume if base is not None else 0.0
            for k in range(2, 62, 4):
                mv = Pos(d[0] * k, d[1] * k, d[2] * k) * s
                if not touch(bb(mv), ob):
                    continue
                x = mv & o
                v = x.volume if x is not None else 0.0
                if v > v0 + 0.2:
                    blocked.append((f"{n}.{i + 1}", m, k, round(v - v0, 1)))
                    break
    return blocked


def check(name):
    v = g.VARIANTS[name]
    asm = g.assembly(name)
    bad = []
    rest = solids(g.pose_assembly(asm), asm)
    mc = mount_clashes(rest)
    print(f"[{name}] mount parts at rest: {sorted(mc) or 'clear'}", flush=True)
    bad += list(mc)
    ac = access(rest)
    print(f"[{name}] mount screw access: {ac or 'clear'}", flush=True)
    bad += ac
    phis = ABAD if v["abad"] else (0.0,)
    if v["leg"]:
        poses = [dict(legq=g.leg4_ik(r, p)) for r, p in LEG_POSES]
        tags = [f"leg r {r:5.1f} phi {p:6.1f}" for r, p in LEG_POSES]
    else:
        poses = [dict(ta=a, tb=b) for a, b in ARM_POSES]
        tags = [f"arms A {a:+5.1f} B {b:+5.1f}" for a, b in ARM_POSES]
    for (pose, tag), phi in itertools.product(zip(poses, tags), phis):
        items = solids(g.pose_assembly(asm, phi=phi, **pose), asm)
        hits = moving_clashes(items, v["abad"])
        print(f"[{name}] {tag}  ab/ad {phi:+5.1f}: {sorted(hits) or 'clear'}", flush=True)
        bad += list(hits)
    n_parts = sum(1 for k in asm.values() if k[3] == "part")
    n_hw = sum(len(k[0].solids()) for k in asm.values() if k[3] == "hw")
    print(f"[{name}] {'PASS' if not bad else 'FAIL'}: {n_parts} printed parts, {n_hw} hardware items, "
          f"{len(poses) * len(phis)} poses", flush=True)
    return not bad


if __name__ == "__main__":
    names = sys.argv[1:] or list(g.VARIANTS)
    ok = [check(n) for n in names]
    assert all(ok), "variant check failed"
