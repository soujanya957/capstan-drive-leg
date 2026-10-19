"""Tessellate the eight builds for make_videos.py.

Writes <out>/<file>.stl (a part shared by several builds is written once) and <out>/<build>.json:
label -> {file, part, color, group, abad}. Motors are split into rotor, stator and magnet holder so
the rotors can spin; the frame's inserts and screws are split one per body, so the film can fit them one
by one (back-plate ones are labelled _back_, the wall ones _front_).
Run:  python export_variants.py <out_dir> [build ...]        (needs build123d)
"""
import json
import os
import sys

import numpy as np
from build123d import CenterOf, Pos

sys.path.insert(0, os.path.dirname(__file__))
import generate as g
from export_meshes import write_stl

EACH = ("FST_Insert_posts", "FST_Bolt_posts", "FST_Screw_posts_back")     # one label per body, named by end
MID_Y = 50.0                                                            # between motor plate and front wall
KEEP_WHOLE = ("REF_Bearing", "Hip_Bearing", "Abad_Bearing", "Leg4_HW_knee_brg", "Leg4_HW_crank_brg", "Leg4_HW_lever_brg")


def fit_info(name, body, others):
    """How the film fits one hardware body: kind, axis and the side it comes in from (unit vector, model
    frame). The axis is the body's odd principal direction (a screw's length, a nut's thickness). Screws
    come in from their head's side; nuts, washers, inserts and spacers from whichever end along the axis
    is clear of the printed and reference parts (swept in 4 mm steps)."""
    pts = np.array([tuple(v) for v in body.tessellate(0.2)[0]])
    c = pts.mean(axis=0)
    w, V = np.linalg.eigh(np.cov((pts - c).T))
    odd = int(np.argmax([abs(w[i] - np.delete(w, i).mean()) for i in range(3)]))
    axis = V[:, odd]
    axis = np.where(abs(axis) > 0.999, np.round(axis), axis)
    along = (pts - c) @ axis
    L = along.max() - along.min()
    D = np.linalg.norm((pts - c) - np.outer(along, axis), axis=1).max() * 2
    com = np.array(tuple(body.center(CenterOf.MASS)))
    mid = c + axis * (along.max() + along.min()) / 2
    lname = name.lower()
    if "nsert" in name:
        kind = "insert"
    elif L / D > 1.6 or (L / D > 0.9 and any(k in lname for k in ("screw", "bolt", "stator", "rotor"))):
        kind = "screw"
    elif L < 2.0:
        kind = "washer"
    elif L / D < 0.75:
        kind = "nut"
    else:
        kind = "other"
    if kind == "screw" and abs((com - mid) @ axis) > 0.03 * L:
        d = axis * np.sign((com - mid) @ axis)
    else:
        cost = []
        for sgn in (1.0, -1.0):
            v = 0.0
            for k in range(4, 29, 4):
                mv = Pos(*(sgn * k * axis)) * body
                mb = mv.bounding_box()
                for o, ob in others:
                    if mb.min.X < ob.max.X and ob.min.X < mb.max.X and mb.min.Y < ob.max.Y and ob.min.Y < mb.max.Y and \
                            mb.min.Z < ob.max.Z and ob.min.Z < mb.max.Z:
                        x = mv & o
                        v += x.volume if x is not None else 0.0
            cost.append(v)
        d = axis * (1.0 if cost[0] <= cost[1] else -1.0)
    return {"kind": kind, "dir": [float(v) for v in d], "len": float(L), "mid": [float(v) for v in mid]}


def pair_with_screws(meta):
    """Make inserts, nuts and washers agree with the screw through them: an insert goes in from the side
    the screw comes from, a nut from the screw's tip end, a washer from the end it sits at."""
    screws = [(np.array(v["fit"]["mid"]), np.array(v["fit"]["dir"]), v["fit"]["len"]) for v in meta.values()
              if "fit" in v and v["fit"]["kind"] == "screw"]
    for v in meta.values():
        f = v.get("fit")
        if not f or f["kind"] not in ("insert", "nut", "washer"):
            continue
        c, d = np.array(f["mid"]), np.array(f["dir"])
        reverse = "shin_inserts" in v.get("part", "")              # v9.1: pressed in from the back plate's outer face
        best = None
        for m, sd, L in screws:
            if abs(abs(sd @ d) - 1.0) > 0.01:
                continue
            rel = c - m
            along = rel @ sd
            if np.linalg.norm(rel - along * sd) < 1.5 and abs(along) < L / 2 + 6.0:
                if best is None or abs(along) < abs(best[1]):
                    best = (sd, along)
        if best is None:
            continue
        sd, along = best
        if f["kind"] == "insert":
            f["dir"] = list(map(float, -sd if reverse else sd))
        elif f["kind"] == "nut":
            f["dir"] = list(map(float, -sd))
        else:
            f["dir"] = list(map(float, sd if along > 0 else -sd))


COLOR = {"desk": "desk", "brass": "brass", "screw": "screw", "steel": "steel", "motor": "motor", "pcb": "pcb",
         "silver": "silver", "yellow": "yellow", "red": "red", "black": "black", "dimgray": "dimgray", "gray": "gray",
         "teal": "teal", "amber": "amber", "violet": "violet"}             # v9.1: the three drums


def main(out, only=()):
    os.makedirs(out, exist_ok=True)
    written = {}                                         # file -> signature
    split = {}

    def put(name, solid, tol, sig):
        file = name
        if file in written and written[file] != sig:
            file = f"{name}__{len([k for k in written if k.startswith(name + '__')]) + 2}"
        if file not in written:
            write_stl(os.path.join(out, f"{file}.stl"), solid, tol, 0.3 if tol < 0.5 else 0.6)
            written[file] = sig
            print(f"  {file}", flush=True)
        return file

    def signature(solid):
        b = solid.bounding_box()
        return (round(solid.volume, 1), round(b.min.X, 1), round(b.min.Y, 1), round(b.min.Z, 1), round(b.max.X, 1),
                round(b.max.Y, 1), round(b.max.Z, 1))

    for w in ("A", "B"):
        split[f"REF_Motor_{w}"] = g.ref_motor_split(w)
    loc_m = g.ab_loc(g.AB_SHIFT)
    split["Abad_Motor"] = {k: loc_m * v for k, v in g.ref_motor_split("B").items()}
    for vname, v in g.VARIANTS.items():
        if only and vname not in only:
            continue
        meta = {}
        asm = g.assembly(vname)
        others = [(s_, s_.bounding_box()) for n_, (so, _, _, k_) in asm.items() if k_ != "hw" and n_ != "REF_Desk"
                  and not n_.startswith("Stand_") for s_ in so.solids()]
        for name, (solid, grp, color, kind) in asm.items():
            if name == "REF_Desk":                                       # the film has its own desk
                continue
            abad = ("fixed" if grp == "world" else "moving") if v["abad"] else None
            if name in split:
                pieces = dict(split[name])
                holder, magnet = sorted(pieces.pop("Magnet_Holder").solids(), key=lambda s_: -s_.volume)
                pieces.update({"Magnet_holder": holder, "Magnet_disc": magnet})
                for piece, part in pieces.items():
                    lab = f"{name}_{piece.lower()}"
                    f = put(lab, part, 0.5, ("motor", lab))
                    meta[lab] = {"file": f, "part": lab, "group": grp, "abad": abad,
                                 "color": {"Rotor": "motor", "Magnet_disc": "magnet", "Magnet_holder": "steel"}.get(piece, "motor_dark")}
                continue
            if name in EACH:
                bodies = sorted(solid.solids(), key=lambda s_: (round(s_.center().X), round(s_.center().Z), s_.center().Y))
                for i, b in enumerate(bodies, 1):
                    end = "" if name.endswith("_back") else "back_" if b.center().Y < MID_Y else "front_"
                    lab = f"{name}_{end}{i}"
                    f = put(lab, b, 0.05, signature(b))
                    meta[lab] = {"file": f, "part": name, "group": grp, "abad": abad,
                                 "color": "brass" if "Insert" in name else "steel",       # stainless frame screws
                                 "fit": fit_info(name, b, others)}
                continue
            if kind == "hw" and len(solid.solids()) > 1 and not name.startswith(KEEP_WHOLE):
                bodies = sorted(solid.solids(), key=lambda s_: (round(s_.center().X), round(s_.center().Z), round(s_.center().Y)))
                for i, b in enumerate(bodies, 1):
                    lab = f"{name}__{i:02d}"
                    f = put(lab, b, 0.05, signature(b))
                    info = fit_info(name, b, others)
                    meta[lab] = {"file": f, "part": name, "group": grp, "abad": abad, "fit": info,
                                 "color": "brass" if info["kind"] == "insert" or "tandoff" in name else COLOR.get(color, "gray")}
                continue
            if kind == "hw" and name.startswith(("REF_Tensioner", "Abad_REF_Tensioner")):
                f = put(name, solid, 0.05, signature(solid))
                meta[name] = {"file": f, "part": name, "group": grp, "abad": abad, "color": COLOR.get(color, "gray"),
                              "fit": fit_info(name, solid, others)}
                continue
            coarse = kind == "ref" and ("moteus" in name)
            tol = 0.5 if coarse else 0.05 if kind == "hw" else 0.15
            f = put(name, solid, tol, signature(solid))
            meta[name] = {"file": f, "part": name, "group": grp, "abad": abad, "color": COLOR.get(color, "gray")}
        pair_with_screws(meta)
        with open(os.path.join(out, f"{vname}.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        print(f"{vname}: {len(meta)} bodies", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
