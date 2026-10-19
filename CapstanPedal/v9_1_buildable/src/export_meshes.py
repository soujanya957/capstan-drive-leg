"""Tessellate every placed part of the v3 assembly into world-frame STL meshes for make_videos.py.

Writes <out>/<label>.stl plus <out>/parts.json (label -> part name, colour).
Run:  python export_meshes.py <out_dir>        (needs build123d)
"""
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import generate as g


def write_stl(path, shape, tol, ang):
    vertices, triangles = shape.tessellate(tol, ang)
    v = np.array([(p.X, p.Y, p.Z) for p in vertices], dtype=np.float32)
    t = np.asarray(triangles, dtype=np.int64)
    tri = v[t]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    keep = np.linalg.norm(n, axis=1) > 1e-9          # MuJoCo rejects degenerate faces
    tri, n = tri[keep], n[keep]
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    with open(path, "wb") as f:
        f.write(b"\0" * 80 + struct.pack("<I", len(tri)))
        rec = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        rec["n"], rec["v"] = n, tri
        f.write(rec.tobytes())
    return len(tri)


def main(out):
    os.makedirs(out, exist_ok=True)
    meta = {}
    for name, (_, placed, color) in g.build().items():
        for i, part in enumerate(placed):
            label = f"{name}_{i + 1}" if len(placed) > 1 else name
            n = write_stl(os.path.join(out, f"{label}.stl"), part, 0.15, 0.3)
            meta[label] = {"part": name, "color": color}
            print(f"  {label:28s} {n:7d} tris", flush=True)
    for w in ("A", "B"):                                  # rotor spins with its small drum
        split = g.ref_motor_split(w)
        holder, magnet = sorted(split.pop("Magnet_Holder").solids(), key=lambda s_: -s_.volume)
        split.update({"Magnet_holder": holder, "Magnet_disc": magnet})
        for piece, part in split.items():
            name = f"REF_Motor_{w}_{piece.lower()}"
            n = write_stl(os.path.join(out, f"{name}.stl"), part, 0.5, 0.6)
            color = {"Rotor": "motor", "Magnet_disc": "magnet", "Magnet_holder": "steel"}.get(piece, "motor_dark")
            meta[name] = {"part": name, "color": color}
            print(f"  {name:28s} {n:7d} tris", flush=True)
    for name, group in g.fasteners().items():          # one body per fastener so each can spin in
        for i, part in enumerate(group.solids()):
            label = f"{name}_{i + 1}"
            write_stl(os.path.join(out, f"{label}.stl"), part, 0.05, 0.3)
            meta[label] = {"part": name, "color": "brass" if "Insert" in name else "steel" if "Nut" in name else "screw"}
        print(f"  {name:28s} x{len(group.solids())}", flush=True)
    for name, part in g.reference().items():
        if name.startswith("REF_Motor"):
            continue
        coarse = "Motor" in name or "moteus" in name    # dense supplier STEP files
        n = write_stl(os.path.join(out, f"{name}.stl"), part, 0.5 if coarse else 0.15, 0.6 if coarse else 0.3)
        meta[name] = {"part": name, "color": "motor" if "Motor" in name else "pcb" if "moteus" in name else "steel"}
        print(f"  {name:28s} {n:7d} tris", flush=True)
    for name, (_, part, color) in g.leg4_parts().items():  # 4-bar leg, used by its leg film only
        write_stl(os.path.join(out, f"{name}.stl"), part, 0.15, 0.3)
        meta[name] = {"part": name, "color": color}
    for name, (part, color) in g.carriage_parts().items():   # rail carriage + MGN12H block (move with the rig)
        write_stl(os.path.join(out, f"{name}.stl"), part, 0.15, 0.3)
        meta[name] = {"part": name, "color": color}
    for v in ("fivebar", "4bar"):                            # leg pins, washers, nuts, shims and bearings
        for name, (part, color, _, _) in g.leg_hardware(v).items():
            write_stl(os.path.join(out, f"{name}.stl"), part, 0.05, 0.3)
            meta[name] = {"part": name, "color": color}
    for name, (part, color) in g.stand_hardware().items():
        write_stl(os.path.join(out, f"{name}.stl"), part, 0.05, 0.3)
        meta[name] = {"part": name, "color": color}
    for name, part in g.stand_parts().items():              # extrusion stand, world coordinates (floor z 0)
        write_stl(os.path.join(out, f"{name}.stl"), part, 0.15, 0.3)
        meta[name] = {"part": name, "color": "silver"}
    for name, (_, part, color) in g.leg_parts().items():   # five-bar leg, used by the leg film only
        n = write_stl(os.path.join(out, f"{name}.stl"), part, 0.15, 0.3)
        meta[name] = {"part": name, "color": color}
        print(f"  {name:28s} {n:7d} tris", flush=True)
    mv, fx = g.ab_parts()                                     # 3-DOF: moving with ab/ad, or fixed in the hip frame
    for group, items in (("moving", mv), ("fixed", fx)):
        for name, (part, color) in items.items():
            if name == "Abad_Motor":
                continue
            write_stl(os.path.join(out, f"{name}.stl"), part, 0.05 if "HW" in name or "REF" in name else 0.15, 0.3)
            meta[name] = {"part": name, "color": color, "abad": group}
    loc_m = g.ab_loc(g.AB_SHIFT)                               # motor C split so its rotor can spin
    split = g.ref_motor_split("B")
    holder, magnet = sorted(split.pop("Magnet_Holder").solids(), key=lambda s_: -s_.volume)
    split.update({"Magnet_holder": holder, "Magnet_disc": magnet})
    for piece, part in split.items():
        name = f"Abad_Motor_{piece.lower()}"
        write_stl(os.path.join(out, f"{name}.stl"), loc_m * part, 0.5, 0.6)
        meta[name] = {"part": name, "abad": "fixed",
                      "color": {"Rotor": "motor", "Magnet_disc": "magnet", "Magnet_holder": "steel"}.get(piece, "motor_dark")}
    with open(os.path.join(out, "parts.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
