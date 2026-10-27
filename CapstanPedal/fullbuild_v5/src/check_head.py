"""v5 head check, as solids: the head's parts and hardware against each other and the body (end frames, plates,
pods at ab/ad 0 and +-AB_LIMIT), with the cradle at each of its seven detents and a right-angle USB-C plug at
either end of the camera. Then the build order's tool paths are checked by check_assembly_quad.py (the film).

Run: python check_head.py
"""
import sys

import generate_quad as q
import head as h
from build123d import Axis

g = q.g


def bb(s):
    b = s.bounding_box()
    return b.min.X, b.min.Y, b.min.Z, b.max.X, b.max.Y, b.max.Z


def vol(a, b):
    """Overlap volume, solid by solid (as check_quad.py)."""
    v = 0.0
    for sa in a.solids():
        for sb in b.solids():
            ba, bb_ = bb(sa), bb(sb)
            if not all(ba[i] < bb_[i + 3] and bb_[i] < ba[i + 3] for i in range(3)):
                continue
            x = sa & sb
            v += x.volume if x is not None else 0.0
    return v


def pods():
    """Each pod's ab/ad-turning parts in the body frame (as check_quad.py)."""
    out = {p: [] for p in q.PODS}
    for n, (s, grp, col, kind) in g.assembly("leg_abad_gantry").items():
        if n.startswith(q.DROP_PREFIX) or grp == "world":
            continue
        for p in q.PODS:
            out[p].append((f"{p}:{n}", q.place(s, p, end=n.startswith(q.END_PREFIX))))
    return out


def ab_tilt(items, p, deg):
    M, t = q.pod_tf(p)
    axis = Axis((0.0, t[1] + M[1, 1] * g.AB_Y, g.AB_Z), (1.0, 0.0, 0.0))
    return [(n, s.rotate(axis, deg * M[1, 1])) for n, s in items]


ALLOWED = (("hd_ins", "End_Frame_R"), ("hd_scr", "hd_ins"), ("hd_scr", "End_Frame_R"), ("hd_scr", "Head_Mount"),
           ("jt_ins", "Head_Mount"), ("jt_scr", "jt_ins"), ("jt_scr", "REF_Jetson_board"), ("jt_scr", "Head_Mount"),
           ("fp_ins", "Head_Mount"), ("fp_scr", "fp_ins"), ("fp_scr", "Head_Face"), ("fp_scr", "Head_Mount"),
           ("cr_ins", "Camera_Cradle"), ("cr_piv", "cr_ins"), ("cr_piv", "Camera_Cradle"), ("cr_piv", "Head_Mount"),
           ("cr_lock", "cr_ins"), ("cr_lock", "Camera_Cradle"), ("cr_lock", "Head_Mount"),
           ("cam_scr", "Camera_Cradle"), ("cam_scr", "REF_ZED_Mini"), ("REF_ZED_Mini", "REF_ZED_Mini_lenses"))
MOVING = ("Camera_Cradle", "REF_ZED", "cr_ins", "cam_scr", "plug")


def allowed(a, b):
    return any((a.startswith(x) and b.startswith(y)) or (b.startswith(x) and a.startswith(y)) for x, y in ALLOWED)


def plugs(tilt):
    """Right-angle USB-C plug envelopes (12 x 22 x 12 mm) at both camera ends, body frame."""
    out = []
    for sy in (-1, 1):
        y0 = sy * h.ZED[0] / 2
        b = q.box(4.0 + 7.0, 4.0 + 19.0, *sorted((y0, y0 + sy * 22.0)), 9.0, 21.0)
        out.append((f"plug_{'R' if sy < 0 else 'L'}", h.body(h.to_cradle(b, tilt))))
    return out


def main():
    body = {n: s for n, (s, c, k) in q.parts().items()}
    head_names = set(h.parts())
    near = {n: s for n, s in body.items() if n not in head_names and n.startswith(("End_Frame_R", "Body_Plate"))}
    fixed_hw = [(n, s) for n, s, *_ in h.hardware() if not n.startswith(("cr_ins", "cam_scr", "cr_lock"))]
    bad = 0
    for tilt in h.TILTS:
        items = [(n, s) for n, (s, c, k) in h.parts(tilt).items()] + plugs(tilt)
        items += [(n, s) for n, s, *_ in h.hardware(tilt) if n.startswith(("cr_ins", "cam_scr", "cr_lock"))]
        items += fixed_hw + list(near.items())
        for i, (a, sa) in enumerate(items):
            for b, sb in items[i + 1:]:
                if tilt != h.TILTS[0] and not (a.startswith(MOVING) or b.startswith(MOVING)):
                    continue                                       # fixed pairs: once
                if allowed(a, b) or (a.startswith("plug") and b.startswith("REF_ZED")) or (a in near and b in near):
                    continue
                v = vol(sa, sb)
                if v > 0.5:
                    bad += 1
                    print(f"  tilt {tilt:2d}: {a} x {b}: {v:.1f} mm3")
        print(f"tilt {tilt:2d} deg: checked", flush=True)
    # the pods at ab/ad 0 and +-AB_LIMIT against the head
    head_all = [(n, s) for n, (s, c, k) in h.parts().items()] + [(n, s) for n, s, *_ in h.hardware()]
    pp = pods()
    for deg in (0.0, q.AB_LIMIT, -q.AB_LIMIT):
        for p in ("LR", "RR"):                                     # the face end's pods
            for n, s in ab_tilt(pp[p], p, deg):
                for a, sa in head_all:
                    v = vol(s, sa)
                    if v > 0.5:
                        bad += 1
                        print(f"  ab/ad {deg:+.0f}: {n} x {a}: {v:.1f} mm3")
    print(f"{bad} clashes" if bad else "pass: the head, its hardware and the cradle at every detent are clear")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
