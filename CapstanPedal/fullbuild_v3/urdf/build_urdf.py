"""URDF of the Capstan quadruped v3 from its CAD (the film meshes generate_quad.py writes: mujoco/meshes).

Links per leg, as the mechanism moves (v9.1's 4-bar is a parallelogram):
  pod_X    the ab/ad-turning module: motor plate, front wall, both motors, gantry, ab/ad drum ...
  thigh_X  output B: drum B, flange B, the thigh, the knee pin
  crank_X  output A: drum A, flange A, the crank, the crank pin. It turns with the shin's ABSOLUTE angle,
           so relative to the thigh it turns with the knee: <mimic joint="knee"/>
  coupler_X  stays parallel to the thigh: relative to the crank, <mimic joint="knee" multiplier="-1"/>
  shin_X   shin, its plates, the foot, the lever pin
The torso is everything else (body, cassette, and the ab/ad motors C, which ride the body).

Joints (axes in the body frame, the robot walks toward -X): abad about +X (+-20 deg), hip about +Y (output B,
+-90), knee about +Y (+-60 from rest: software limits on the robot). Output A = hip + knee (a tendon in the
MuJoCo model, sim_model.py). Joint zero = the CAD's rest pose: thigh 60 deg below horizontal, knee bent 60.

Visual meshes: one STL per link per colour (metres, link frame). Collision: primitives (box body, capsule
links, 23 mm ball feet). Inertia: each part's mesh as a solid of its material's density (motors and the
battery by catalogue mass), summed per link.

Run: python build_urdf.py [mesh_dir]     -> capstan_quad.urdf, meshes/
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
import make_quad_film as qf                                          # noqa: E402

mv = qf.mv
PODS = ("LF", "LR", "RF", "RR")
FOOT_R = 0.023

# g/cm^3 by film colour (printed PETG at ~60 % fill), or a fixed mass in g
DENSITY = {"black": 1.27 * 0.6, "red": 1.27 * 0.6, "teal": 1.27 * 0.6, "amber": 1.27 * 0.6, "violet": 1.27 * 0.6,
           "orange": 1.27 * 0.6, "purple": 1.27 * 0.6, "white": 1.27 * 0.6, "yellow": 1.27 * 0.6,
           "cf": 1.6, "steel": 7.8, "silver": 7.8, "screw": 7.8, "brass": 8.5, "dimgray": 2.7, "pcb": 1.85}
FIXED_G = {"motor": 210.0, "motor_dark": 110.0, "magnet": 2.0, "blue": 290.0}   # EaglePower 8308 ~320 g; 6S 1800 pack
RGBA = {k: (*v[0], 1.0) for k, v in mv.PALETTE.items()}


def mass_props(tri):
    """Volume (mm^3), centroid, and inertia about the centroid (per unit density) of a closed mesh."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    v6 = np.einsum("ij,ij->i", a, np.cross(b, c))
    vol = v6.sum() / 6.0
    if abs(vol) < 1e-9:
        return 0.0, tri.reshape(-1, 3).mean(0), np.zeros((3, 3))
    com = ((a + b + c) * v6[:, None]).sum(0) / (24.0 * vol)
    # second moments about the origin, per tetrahedron (0, a, b, c)
    S = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            S[i, j] = (v6 * (2 * (a[:, i] * a[:, j] + b[:, i] * b[:, j] + c[:, i] * c[:, j])
                             + a[:, i] * b[:, j] + a[:, j] * b[:, i] + a[:, i] * c[:, j] + a[:, j] * c[:, i]
                             + b[:, i] * c[:, j] + b[:, j] * c[:, i])).sum() / 120.0
    sign = 1.0 if vol > 0 else -1.0
    vol, S = abs(vol), S * sign
    S_c = S - vol * np.outer(com, com)
    I = np.trace(S_c) * np.eye(3) - S_c
    return vol, com, I / vol                                         # inertia per unit mass


def link_of(v):
    """Which link a CAD part rides on."""
    if "pod" not in v:
        return "torso"
    kind, rides = qf.classify(v)
    if not rides:
        return "torso"
    p = v["pod"]
    return {"spin:B": "thigh", "joint:knee": "thigh", "spin:A": "crank", "joint:crank": "crank",
            "coupler": "coupler", "shin": "shin", "joint:lever": "shin"}.get(kind, "pod") + "_" + p


def decimate(tri, cell):
    """Vertex clustering: snap vertices to a grid of `cell` (m), merge, drop collapsed and repeated triangles.
    Visual meshes only (inertia comes from the full CAD); CAD hardware is mostly fine thread and fillet facets."""
    v = tri.reshape(-1, 3)
    key = np.floor(v / cell).astype(np.int64)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    sums = np.zeros((len(uk), 3))
    np.add.at(sums, inv, v)
    rep = sums / np.bincount(inv, minlength=len(uk))[:, None]
    f = inv.reshape(-1, 3)
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[keep]
    _, first = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    return rep[f[np.sort(first)]]


HARDWARE = ("screw", "brass", "steel", "silver", "magnet")


def write_stl(path, tri):
    with open(path, "wb") as f:
        f.write(b"\0" * 80 + struct.pack("<I", len(tri)))
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        rec = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", 9), ("a", "<u2")])
        rec["n"], rec["v"] = n, tri.reshape(-1, 9)
        f.write(rec.tobytes())


def frames(meta):
    """Joint points (m, body frame, at rest) per pod: ab/ad axis point, hip, knee, crank pin."""
    out = {}
    for p in PODS:
        k = f"{p}__Motor_Plate"
        D, sh = np.diag(meta[k]["mirror"]), np.array(meta[k]["shift"])
        ab = D @ np.array(mv.AB_C, float) + sh
        hip = np.array([sh[0] + mv.OUT[0], 0.0, mv.OUT[1]])
        knee = np.array([sh[0] + mv.L4_K0[0], 0.0, mv.L4_K0[1]])
        crank = np.array([sh[0] + mv.L4_C0[0], 0.0, mv.L4_C0[1]])
        foot = np.array([sh[0] + mv.L4_F0[0], 0.0, mv.L4_F0[1]])
        out[p] = {n: x / 1000.0 for n, x in (("ab", ab), ("hip", hip), ("knee", knee), ("crank", crank), ("foot", foot))}
    return out


def main(mesh_dir=None):
    import cad
    c = cad.Cad(mesh_dir or cad.MESHES)
    meta, tris = c.meta, c.tris
    F = frames(meta)
    leg_y = {}
    for p in PODS:                                                     # the leg's plane: the shin's mid-y
        ys = tris[f"{p}__Leg4_Shin"].reshape(-1, 3)[:, 1]
        leg_y[p] = (ys.min() + ys.max()) / 2 / 1000.0
    origin = {"torso": np.zeros(3)}                                    # each link frame's origin, body frame (m)
    for p in PODS:
        origin[f"pod_{p}"] = np.array([F[p]["hip"][0], F[p]["ab"][1], F[p]["ab"][2]])
        origin[f"thigh_{p}"] = np.array([F[p]["hip"][0], leg_y[p], F[p]["hip"][2]])
        origin[f"crank_{p}"] = origin[f"thigh_{p}"]
        origin[f"coupler_{p}"] = np.array([F[p]["crank"][0], leg_y[p], F[p]["crank"][2]])
        origin[f"shin_{p}"] = np.array([F[p]["knee"][0], leg_y[p], F[p]["knee"][2]])
    groups, inert = {}, {}
    for k, v in meta.items():
        if k == "Q_Build_Stand":
            continue
        ln = link_of(v)
        t = tris[k] / 1000.0 - origin[ln]                              # m, link frame
        groups.setdefault((ln, v["color"]), []).append(t)
        vol, com, I1 = mass_props(tris[k] / 10.0)                      # cm: density in g/cm^3
        m = FIXED_G.get(v["color"], vol * DENSITY.get(v["color"], 1.2)) / 1000.0     # kg
        com_m = com / 100.0 - origin[ln]
        I = I1 * m * 1e-4                                               # kg m^2
        M, C, J = inert.get(ln, (0.0, np.zeros(3), np.zeros((3, 3))))
        inert[ln] = (M + m, C + m * com_m, J + I + m * (np.dot(com_m, com_m) * np.eye(3) - np.outer(com_m, com_m)))
    md = HERE / "meshes"
    md.mkdir(exist_ok=True)
    for f in md.glob("*.stl"):
        f.unlink()
    visuals = {}
    n_full = n_small = 0
    for (ln, col), ts in sorted(groups.items()):
        name = f"{ln}__{col}.stl"
        full = np.concatenate(ts)
        small = decimate(full, 0.0010 if col in HARDWARE else 0.0006)
        write_stl(md / name, small.astype(np.float32))
        n_full, n_small = n_full + len(full), n_small + len(small)
        visuals.setdefault(ln, []).append((name, col))
    print({ln: round(v[0], 3) for ln, v in inert.items()}, "total", round(sum(v[0] for v in inert.values()), 2), "kg")

    def inertial(ln):
        m, mc, J = inert[ln]
        com = mc / m
        Jc = J - m * (np.dot(com, com) * np.eye(3) - np.outer(com, com))   # about the com
        w, _ = np.linalg.eigh(Jc)
        Jc = Jc + np.eye(3) * max(0.0, 1e-6 - w.min())                    # keep it positive definite
        return (f'    <inertial><origin xyz="{com[0]:.5f} {com[1]:.5f} {com[2]:.5f}"/><mass value="{m:.4f}"/>'
                f'<inertia ixx="{Jc[0, 0]:.6g}" ixy="{Jc[0, 1]:.6g}" ixz="{Jc[0, 2]:.6g}" iyy="{Jc[1, 1]:.6g}" '
                f'iyz="{Jc[1, 2]:.6g}" izz="{Jc[2, 2]:.6g}"/></inertial>\n')

    def vis(ln):
        return "".join(f'    <visual><geometry><mesh filename="meshes/{n}"/></geometry>'
                       f'<material name="{col}"/></visual>\n' for n, col in visuals.get(ln, []))

    def capsule(a, b, r):
        a, b = np.asarray(a), np.asarray(b)
        d = b - a
        L = float(np.linalg.norm(d))
        c_ = (a + b) / 2
        pitch = float(np.arctan2(d[0], d[2]))                          # cylinder along z, turned about y
        return (f'    <collision><origin xyz="{c_[0]:.5f} {c_[1]:.5f} {c_[2]:.5f}" rpy="0 {pitch:.5f} 0"/>'
                f'<geometry><cylinder radius="{r}" length="{L:.5f}"/></geometry></collision>\n')

    links, joints = [], []
    links.append('  <link name="torso">\n' + inertial("torso") + vis("torso") +
                 '    <collision><origin xyz="0 0 -0.0325"/><geometry><box size="0.69 0.26 0.22"/></geometry></collision>\n'
                 '  </link>\n')
    for p in PODS:
        sx, sy = (1 if p[1] == "F" else -1), (1 if p[0] == "L" else -1)
        o = origin
        pod_box_c = np.array([sx * 0.19, sy * 0.10, -0.03]) - o[f"pod_{p}"]
        links.append(f'  <link name="pod_{p}">\n' + inertial(f"pod_{p}") + vis(f"pod_{p}") +
                     f'    <collision><origin xyz="{pod_box_c[0]:.4f} {pod_box_c[1]:.4f} {pod_box_c[2]:.4f}"/>'
                     f'<geometry><box size="0.26 0.10 0.17"/></geometry></collision>\n  </link>\n')
        knee_t = F[p]["knee"] - F[p]["hip"]
        links.append(f'  <link name="thigh_{p}">\n' + inertial(f"thigh_{p}") + vis(f"thigh_{p}") +
                     capsule((0, 0, 0), (knee_t[0], 0, knee_t[2]), 0.014) + '  </link>\n')
        links.append(f'  <link name="crank_{p}">\n' + inertial(f"crank_{p}") + vis(f"crank_{p}") + '  </link>\n')
        links.append(f'  <link name="coupler_{p}">\n' + inertial(f"coupler_{p}") + vis(f"coupler_{p}") + '  </link>\n')
        foot_s = F[p]["foot"] - F[p]["knee"]
        links.append(f'  <link name="shin_{p}">\n' + inertial(f"shin_{p}") + vis(f"shin_{p}") +
                     capsule((0, 0, 0), (foot_s[0] * 0.85, 0, foot_s[2] * 0.85), 0.012) +
                     f'    <collision name="foot_{p}"><origin xyz="{foot_s[0]:.5f} 0 {foot_s[2]:.5f}"/>'
                     f'<geometry><sphere radius="{FOOT_R}"/></geometry></collision>\n  </link>\n')

        def j(name, parent, child, xyz, axis, lo, hi, mimic=None, eff=31.5, vel=23.3):
            m_ = f'    <mimic joint="{mimic[0]}" multiplier="{mimic[1]}"/>\n' if mimic else ""
            joints.append(f'  <joint name="{name}" type="revolute">\n    <parent link="{parent}"/><child link="{child}"/>\n'
                          f'    <origin xyz="{xyz[0]:.5f} {xyz[1]:.5f} {xyz[2]:.5f}"/><axis xyz="{axis}"/>\n'
                          f'    <limit lower="{np.radians(lo):.5f}" upper="{np.radians(hi):.5f}" effort="{eff}" velocity="{vel}"/>\n'
                          f'{m_}  </joint>\n')
        rel = lambda a, b: o[a] - o[b]
        j(f"{p}_abad", "torso", f"pod_{p}", o[f"pod_{p}"], "1 0 0", -20, 20)
        j(f"{p}_hip", f"pod_{p}", f"thigh_{p}", rel(f"thigh_{p}", f"pod_{p}"), "0 1 0", -90, 90)
        j(f"{p}_knee", f"thigh_{p}", f"shin_{p}", rel(f"shin_{p}", f"thigh_{p}"), "0 1 0", -60, 60)
        j(f"{p}_crank", f"thigh_{p}", f"crank_{p}", rel(f"crank_{p}", f"thigh_{p}"), "0 1 0", -60, 60, (f"{p}_knee", 1))
        j(f"{p}_coupler", f"crank_{p}", f"coupler_{p}", rel(f"coupler_{p}", f"crank_{p}"), "0 1 0", -60, 60, (f"{p}_knee", -1))
    mats = "".join(f'  <material name="{c_}"><color rgba="{r:.3f} {g:.3f} {b:.3f} {a:.1f}"/></material>\n'
                   for c_, (r, g, b, a) in sorted(RGBA.items()) if any(c_ == col for v in visuals.values() for _, col in v))
    urdf = (f'<?xml version="1.0"?>\n<!-- Capstan quadruped v3 (v9.1 legs): generated by build_urdf.py from the CAD -->\n'
            f'<robot name="capstan_quad">\n{mats}{"".join(links)}{"".join(joints)}</robot>\n')
    (HERE / "capstan_quad.urdf").write_text(urdf)
    json.dump({p: {k: v.tolist() for k, v in F[p].items()} | {"leg_y": leg_y[p]} for p in PODS},
              open(HERE / "frames.json", "w"), indent=1)
    print(f"capstan_quad.urdf: {len(links)} links, {len(joints)} joints, {sum(len(v) for v in visuals.values())} visual meshes, "
          f"{n_small} triangles (decimated from {n_full})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
