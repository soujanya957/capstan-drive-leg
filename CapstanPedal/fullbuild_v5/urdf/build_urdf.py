"""URDF of the Capstan quadruped v5 from its CAD (the film meshes generate_quad.py writes: mujoco/meshes).

Links per leg, as the mechanism moves:
  pod_X      the ab/ad-turning module: motor plate, front wall, both motor stators, gantry, ab/ad drum ...
  thigh_X    output B (joint X_hip): drum B, flange B, the thigh, the knee pin
  crank_X    output A (joint X_crank, ABSOLUTE, coaxial with the hip): drum A, flange A, the crank
  coupler_X  on the crank pin (passive joint X_coupler)
  shin_X     on the knee (passive joint X_knee): shin, plates, foot, lever pin
  rotorA_X, rotorB_X   motor A and B rotors with their small drums: <mimic> the output they drive, x -9
                       (the 9:1 capstan; the rotor turns the other way)
  rotorC_X   motor C's rotor and small drum (on the body, about X): <mimic> X_abad, x -9
The 4-bar is a closed loop (crank - coupler - shin - thigh, a parallelogram): URDF cannot close it, so the
coupler's far end and the shin's lever pin are listed in frames.json ("lever_on_coupler", "lever_on_shin")
and the MuJoCo model (mujoco/model.py) pins them together. Driven joints: X_abad (motor C), X_hip (motor B),
X_crank (motor A); 31.5 N*m each.

Joint zero = the CAD's rest pose: thigh 60 deg below horizontal, shin 120 (knee bent 60). Axes in the body
frame (the robot walks toward -X): abad +X (+-AB_LIMIT: v4 12 deg), hip and crank +Y (+-90: the capstans' travel),
knee +Y (+-60 from rest: software limits on the robot).

Visual meshes: one OBJ per link per colour (metres, link frame, normals creased at 30 deg; hardware simplified). Collision:
primitives (box body, capsule links, 23 mm ball feet). Inertia: each part's mesh as a solid of its
material's density (motors and battery by catalogue mass), summed per link; the rotors' mass is booked to
the pod (or body) that carries them, and their inertia to the driven joints as reflected inertia in MuJoCo.

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
FIXED_G = {"motor": 210.0, "motor_dark": 110.0, "magnet": 2.0, "blue": 755.0,   # EaglePower 8308 ~320 g; v5: 6S 5000 pack (v4: 1800, 290 g)
           "zed": 63.0, "jetson": 130.0, "port": 25.0, "buck": 15.0}          # v5 head: ZED Mini; Orin Nano module + heatsink, ports; converter
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
    p = v["pod"]
    if kind == "abadspin":
        return f"rotorC_{p}"
    if not rides:
        return "torso"
    return {"spin:B": "thigh", "joint:knee": "thigh", "spin:A": "crank", "joint:crank": "crank",
            "coupler": "coupler", "shin": "shin", "joint:lever": "shin",
            "spin:mA": "rotorA", "spin:mB": "rotorB"}.get(kind, "pod") + "_" + p


def mass_link(ln):
    """Where a link's mass is booked: the rotors' on what carries them."""
    if ln.startswith(("rotorA", "rotorB")):
        return "pod_" + ln.split("_")[1]
    if ln.startswith("rotorC"):
        return "torso"
    return ln


def weld(tri, tol=1e-6):
    """Triangle soup (n, 3, 3) -> (vertices, faces), merging coincident corners."""
    v = tri.reshape(-1, 3)
    uk, inv = np.unique(np.round(v / tol).astype(np.int64), axis=0, return_inverse=True)
    pts = np.zeros((len(uk), 3))
    pts[inv.ravel()] = v
    f = inv.ravel().reshape(-1, 3)
    return pts, f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]


def is_hardware(v):
    """Screws, inserts, nuts, pins, bearings: simplified for display. Printed parts and drums never are."""
    n = v.get("base", v.get("part", ""))
    if "Drum" in n or "Flange" in n:
        return False
    return v["color"] in ("screw", "brass", "steel", "silver", "magnet") or n.startswith(("FST_", "REF_Bearing")) or "_HW_" in n


def simplify(tri, keep):
    """Quadric edge collapse (fast_simplification) to `keep` of the triangles; flat faces stay flat."""
    import fast_simplification
    pts, f = weld(tri)
    if len(f) < 200:
        return tri
    p2, f2 = fast_simplification.simplify(pts.astype(np.float32), f.astype(np.int32), target_reduction=1.0 - keep)
    return p2[f2]


def write_obj(path, tri, crease=30.0):
    """OBJ with per-corner normals: averaged over the faces round a vertex that are within `crease` degrees of
    the corner's own face (curved surfaces shade smooth, edges and flat faces stay crisp: no streaks round holes)."""
    pts, f = weld(tri)
    fn = np.cross(pts[f[:, 1]] - pts[f[:, 0]], pts[f[:, 2]] - pts[f[:, 0]])
    area = np.linalg.norm(fn, axis=1)
    keep = area > 1e-14
    f, fn, area = f[keep], fn[keep], area[keep]
    nf = fn / area[:, None]
    nv, nfc = len(pts), len(f)
    corner_v = f.ravel()
    corner_f = np.repeat(np.arange(nfc), 3)
    order = np.argsort(corner_v, kind="stable")
    cnt = np.bincount(corner_v, minlength=nv)
    start = np.concatenate([[0], np.cumsum(cnt)])
    K = int(cnt.max())
    inc = -np.ones((nv, K), np.int64)                                # faces round each vertex
    rank = np.arange(len(order)) - start[corner_v[order]]
    inc[corner_v[order], rank] = corner_f[order]
    cosc = np.cos(np.radians(crease))
    normals = np.zeros((nfc * 3, 3))
    for c0 in range(0, nfc * 3, 300000):                             # in chunks (memory)
        cv, cf = corner_v[c0:c0 + 300000], corner_f[c0:c0 + 300000]
        g = inc[cv]                                                  # (m, K)
        valid = g >= 0
        gn = nf[np.where(valid, g, 0)]                               # (m, K, 3)
        dots = np.einsum("mkj,mj->mk", gn, nf[cf])
        w = np.where(valid & (dots > cosc), area[np.where(valid, g, 0)], 0.0)
        n = np.einsum("mk,mkj->mj", w, gn)
        normals[c0:c0 + 300000] = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    nq = np.round(normals, 4)
    un, ninv = np.unique(nq, axis=0, return_inverse=True)
    ninv = ninv.ravel().reshape(-1, 3) + 1
    with open(path, "w") as fh:
        fh.write("".join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in pts))
        fh.write("".join(f"vn {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in un))
        ff = f + 1
        fh.write("".join(f"f {a}//{na} {b}//{nb} {c}//{nc}\n" for (a, b, c), (na, nb, nc) in zip(ff, ninv)))


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
        lever = np.array([sh[0] + mv.L4_D0[0], 0.0, mv.L4_D0[1]])
        mA = np.array([sh[0] + mv.MA[0], ab[1], mv.MA[1]])
        mB = np.array([sh[0] + mv.MB[0], ab[1], mv.MB[1]])
        mC = D @ (np.array(mv.AB_C, float) + np.array([0.0, 0.0, 84.0])) + sh
        out[p] = {n: x / 1000.0 for n, x in (("ab", ab), ("hip", hip), ("knee", knee), ("crank", crank), ("foot", foot),
                                             ("lever", lever), ("mA", mA), ("mB", mB), ("mC", mC))}
    return out


def main(mesh_dir=None):
    import cad
    c = cad.Cad(mesh_dir or cad.MESHES)
    meta, tris = c.meta, c.tris
    global AB_LIMIT
    AB_LIMIT = float(c.layout.get("ab_limit", 20.0))
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
        origin[f"rotorA_{p}"] = F[p]["mA"]
        origin[f"rotorB_{p}"] = F[p]["mB"]
        origin[f"rotorC_{p}"] = F[p]["mC"]
        origin[f"coupler_{p}"] = np.array([F[p]["crank"][0], leg_y[p], F[p]["crank"][2]])
        origin[f"shin_{p}"] = np.array([F[p]["knee"][0], leg_y[p], F[p]["knee"][2]])
    groups, inert = {}, {}
    for k, v in meta.items():
        if k == "Q_Build_Stand":
            continue
        ln = link_of(v)
        t = tris[k] / 1000.0 - origin[ln]                              # m, link frame
        groups.setdefault((ln, v["color"]), []).append(simplify(t, 0.3) if is_hardware(v) else t)
        vol, com, I1 = mass_props(tris[k] / 10.0)                      # cm: density in g/cm^3
        rho = 1.2 if k.endswith("Leg4_Foot") else DENSITY.get(v["color"], 1.2)  # v5: the feet are TPU (v4 booked them at 2.7)
        m = FIXED_G.get(v["color"], vol * rho) / 1000.0                  # kg
        mln = mass_link(ln)
        com_m = com / 100.0 - origin[mln]
        I = I1 * m * 1e-4                                               # kg m^2
        M, C, J = inert.get(mln, (0.0, np.zeros(3), np.zeros((3, 3))))
        inert[mln] = (M + m, C + m * com_m, J + I + m * (np.dot(com_m, com_m) * np.eye(3) - np.outer(com_m, com_m)))
    md = HERE / "meshes"
    md.mkdir(exist_ok=True)
    for f in list(md.glob("*.stl")) + list(md.glob("*.obj")):
        f.unlink()
    visuals = {}
    n_full = n_small = 0
    for (ln, col), ts in sorted(groups.items()):
        name = f"{ln}__{col}.obj"
        full = np.concatenate(ts)
        write_obj(md / name, full)
        n_small += len(full)
        visuals.setdefault(ln, []).append((name, col))
    print({ln: round(v[0], 3) for ln, v in inert.items()}, "total", round(sum(v[0] for v in inert.values()), 2), "kg")

    def inertial(ln):
        if ln not in inert:                                             # a rotor: its mass is on its carrier
            return '    <inertial><mass value="0.001"/><inertia ixx="1e-7" ixy="0" ixz="0" iyy="1e-7" iyz="0" izz="1e-7"/></inertial>\n'
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
    hb = np.array(c.layout["head"]["box"])                             # v5: the head (and its camera) collides too
    hc, hs = (hb[0] + hb[1]) / 2, hb[1] - hb[0]
    bb_ = np.array(c.layout["belly"]["box"])                           # v5: the belly pod (its cover is the skid)
    bc, bs = (bb_[0] + bb_[1]) / 2, bb_[1] - bb_[0]
    links.append('  <link name="torso">\n' + inertial("torso") + vis("torso") +
                 '    <collision><origin xyz="0 0 -0.0325"/><geometry><box size="0.69 0.26 0.22"/></geometry></collision>\n'
                 f'    <collision name="head"><origin xyz="{hc[0]:.4f} {hc[1]:.4f} {hc[2]:.4f}"/><geometry><box size="{hs[0]:.4f} {hs[1]:.4f} {hs[2]:.4f}"/></geometry></collision>\n'
                 f'    <collision name="belly"><origin xyz="{bc[0]:.4f} {bc[1]:.4f} {bc[2]:.4f}"/><geometry><box size="{bs[0]:.4f} {bs[1]:.4f} {bs[2]:.4f}"/></geometry></collision>\n'
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
        for r in "ABC":
            links.append(f'  <link name="rotor{r}_{p}">\n' + inertial(f"rotor{r}_{p}") + vis(f"rotor{r}_{p}") + '  </link>\n')
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
        j(f"{p}_abad", "torso", f"pod_{p}", o[f"pod_{p}"], "1 0 0", -AB_LIMIT, AB_LIMIT)
        j(f"{p}_hip", f"pod_{p}", f"thigh_{p}", rel(f"thigh_{p}", f"pod_{p}"), "0 1 0", -90, 90)
        j(f"{p}_crank", f"pod_{p}", f"crank_{p}", rel(f"crank_{p}", f"pod_{p}"), "0 1 0", -90, 90)
        j(f"{p}_knee", f"thigh_{p}", f"shin_{p}", rel(f"shin_{p}", f"thigh_{p}"), "0 1 0", -60, 60, eff=0)
        j(f"{p}_coupler", f"crank_{p}", f"coupler_{p}", rel(f"coupler_{p}", f"crank_{p}"), "0 1 0", -60, 60, eff=0)
        j(f"{p}_rotorA", f"pod_{p}", f"rotorA_{p}", rel(f"rotorA_{p}", f"pod_{p}"), "0 1 0", -1e4, 1e4, (f"{p}_crank", -9), eff=3.5, vel=210)
        j(f"{p}_rotorB", f"pod_{p}", f"rotorB_{p}", rel(f"rotorB_{p}", f"pod_{p}"), "0 1 0", -1e4, 1e4, (f"{p}_hip", -9), eff=3.5, vel=210)
        j(f"{p}_rotorC", "torso", f"rotorC_{p}", o[f"rotorC_{p}"], "1 0 0", -1e4, 1e4, (f"{p}_abad", -9), eff=3.5, vel=210)
    mats = "".join(f'  <material name="{c_}"><color rgba="{r:.3f} {g:.3f} {b:.3f} {a:.1f}"/></material>\n'
                   for c_, (r, g, b, a) in sorted(RGBA.items()) if any(c_ == col for v in visuals.values() for _, col in v))
    urdf = (f'<?xml version="1.0"?>\n<!-- Capstan quadruped v5 (v9.1 legs, a head with a Jetson and a stereo camera): generated by build_urdf.py from the CAD -->\n'
            f'<robot name="capstan_quad">\n{mats}{"".join(links)}{"".join(joints)}</robot>\n')
    (HERE / "capstan_quad.urdf").write_text(urdf)
    for p in PODS:                                                      # the 4-bar's loop closure, per leg
        F[p]["lever_on_coupler"] = F[p]["lever"] - F[p]["crank"]
        F[p]["lever_on_shin"] = F[p]["lever"] - F[p]["knee"]
    json.dump({p: {k: v.tolist() for k, v in F[p].items()} | {"leg_y": leg_y[p]} for p in PODS} | {"head": c.layout["head"]},
              open(HERE / "frames.json", "w"), indent=1)
    print(f"capstan_quad.urdf: {len(links)} links, {len(joints)} joints, {sum(len(v) for v in visuals.values())} visual meshes, "
          f"{n_small} triangles (hardware simplified to 30 %; printed parts and drums at full detail)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
