"""Quadruped film (fullbuild_v4): the electronics cassette built on the bench; the body built round the four
ab/ad drives (dock, centre bulkheads with motors C, ab/ad drums, inner bulkheads); the four v9.1 pods hung on
their hubs and screwed into the hubs' inserts; the end frames over the trunnions; the top plate; the cassette
dropped in from above and the lid's thumb screws. Then the robot stands up, trots and jumps (kinematics only).
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V62 = (ROOT.parent / "v9_1_buildable" / "src").resolve()       # the v9.1 leg module
sys.path.insert(0, str(V62))
sys.path.insert(0, str(HERE))
import generate_quad as gq                                           # noqa: E402  (sets the legs' length first)
import make_videos as mv                                            # noqa: E402

mv.PALETTE.update({"cf": ((0.09, 0.09, 0.10), 0.55, 0.7, 0.04), "blue": ((0.12, 0.30, 0.85), 0.35, 0.5, 0.0),
                   "purple": ((0.36, 0.20, 0.55), 0.3, 0.4, 0.0)})
PODS = ("LF", "LR", "RF", "RR")
FOOT_R = mv.L4_FOOT_R
HIP_Z = mv.OUT[1]


# ------------------------------------------------------------------------------------------------
# Scene
# ------------------------------------------------------------------------------------------------
def mat_name(k, meta):
    return f"m_{meta[k]['pod']}_{meta[k]['color']}" if "pod" in meta[k] else f"m_{k}"


def build_model(meta):
    assets, bodies, mats = [], [], {}
    for k, info in meta.items():
        mn = mat_name(k, meta)
        if mn not in mats:
            rgb, spec, shin, refl = mv.PALETTE[info["color"]]
            mats[mn] = (f'<material name="{mn}" rgba="{rgb[0]} {rgb[1]} {rgb[2]} 1" specular="{spec}" '
                        f'shininess="{shin}" reflectance="{refl}"/>')
        assets.append(f'<mesh name="{k}" file="{info["file"]}.stl" inertia="shell"/>')
        bodies.append(f'<body name="{k}" mocap="true"><geom type="mesh" mesh="{k}" material="{mn}" '
                      f'contype="0" conaffinity="0" mass="0.01"/></body>')
    th = mv.THEMES["dark"]
    xml = f"""
<mujoco model="capstan_quadruped">
  <compiler meshdir="/" angle="radian"/>
  <statistic extent="900" center="0 0 200"/>
  <visual>
    <global offwidth="{mv.W}" offheight="{mv.H}" fovy="{mv.FOVY}"/>
    <quality shadowsize="8192" offsamples="8"/>
    <map znear="0.002" zfar="40" shadowclip="1.6" shadowscale="0.5" haze="0.15"/>
    <headlight ambient="0.32 0.32 0.33" diffuse="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="{th["sky"][0]}" rgb2="{th["sky"][1]}" width="64" height="1024"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.18 0.2 0.23" rgb2="0.12 0.13 0.15" mark="edge" markrgb="0.3 0.33 0.36" width="512" height="512"/>
    <material name="floor" texture="grid" texrepeat="40 40" reflectance="0.2"/>
    {''.join(mats.values())}
    {''.join(assets)}
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 1" dir="-0.35 -0.45 -0.82" diffuse="0.55 0.55 0.53" specular="0.25 0.25 0.25" castshadow="true"/>
    <light directional="true" pos="0 0 1" dir="0.6 0.3 -0.4" diffuse="0.18 0.19 0.22" specular="0 0 0" castshadow="false"/>
    <light directional="true" pos="0 0 1" dir="0.0 -0.8 0.3" diffuse="0.10 0.10 0.11" specular="0 0 0" castshadow="false"/>
    <geom name="floor" type="plane" size="6000 6000 1" pos="0 0 {th["floor"]}" material="floor" contype="0" conaffinity="0"/>
    {''.join(bodies)}
  </worldbody>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def mesh_data(model, meta):
    verts, tris = {}, {}
    for k in meta:
        gid = model.body_geomadr[model.body(k).id]
        mid = model.geom_dataid[gid]
        v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        f = model.mesh_face[model.mesh_faceadr[mid]:model.mesh_faceadr[mid] + model.mesh_facenum[mid]]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.geom_quat[gid])
        w = v @ R.reshape(3, 3).T + model.geom_pos[gid]
        if "shift" in meta[k]:
            w = w + np.array(meta[k]["shift"])
        verts[k] = np.array([w.min(0), w.max(0), (w.min(0) + w.max(0)) / 2])
        tris[k] = w[f]
    return verts, tris


# ------------------------------------------------------------------------------------------------
# Pod poses (pod-local, unmirrored), as make_videos.render does for the single leg
# ------------------------------------------------------------------------------------------------
SPIN_OF_PREFIX = [(p, grp) for grp, ps in mv.SPIN.items() for p in ps]
CENTERS = {"A": np.array([mv.OUT[0], 0, mv.OUT[1]]), "B": np.array([mv.OUT[0], 0, mv.OUT[1]]),
           "mA": np.array([mv.MA[0], 0, mv.MA[1]]), "mB": np.array([mv.MB[0], 0, mv.MB[1]])}


def xz_rot(angle, pivot0, pivot):
    dp, qq = mv.rot_xz(angle, np.array([pivot0[0], 0.0, pivot0[1]]))
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, qq)
    return R.reshape(3, 3), dp + np.array([pivot[0] - pivot0[0], 0.0, pivot[1] - pivot0[1]])


def classify(v):
    base = v["base"]
    rides = v.get("abad") != "fixed"
    if base in ("Leg4_Shin", "Leg4_Foot") or base.startswith("Leg4_Shin_Plate"):
        return "shin", rides
    if base == "Leg4_Coupler":
        return "coupler", rides
    if base.startswith("Leg4_HW_"):
        j = v.get("part", base)[len("Leg4_HW_"):].rsplit("_", 1)[0]
        return ("shin" if j in ("foot", "shin") else f"joint:{j}"), rides
    if base.startswith(mv.ABAD_SPIN):
        return "abadspin", rides
    for p, grp in SPIN_OF_PREFIX:
        if base.startswith(p):
            return f"spin:{grp}", rides
    return "fixed", rides


def leg_state(qt, qs, ab):
    th = {"A": math.radians(qs - mv.L4_Q0["shin"]), "B": math.radians(qt - mv.L4_Q0["thigh"])}
    ang = {"A": th["A"], "B": th["B"], "mA": -mv.RATIO * th["A"], "mB": -mv.RATIO * th["B"]}
    pts = mv.leg4_points(qt, qs)
    st = {"th": th, "ab": ab, "Rab": mv.rot_x(ab)}
    for grp, a in ang.items():
        dp, qq = mv.rot_xz(a, CENTERS[grp])
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, qq)
        st[f"spin:{grp}"] = (R.reshape(3, 3), dp)
    st["shin"] = xz_rot(math.radians(qs - mv.L4_Q0["shin"]), mv.L4_K0, pts[0])
    st["coupler"] = xz_rot(math.radians(qt - mv.L4_Q0["thigh"]), mv.L4_C0, pts[2])
    for j, idx in (("knee", 0), ("crank", 2), ("lever", 3)):
        p, p0 = pts[idx], (mv.L4_K0, mv.L4_F0, mv.L4_C0, mv.L4_D0)[idx]
        st[f"joint:{j}"] = (np.eye(3), np.array([p[0] - p0[0], 0.0, p[1] - p0[1]]))
    Rm, cm = mv.rot_x(-mv.RATIO * ab), mv.AB_C + np.array([0.0, 0.0, 84.0])
    st["abadspin"] = (Rm, cm - Rm @ cm)
    st["fixed"] = (np.eye(3), np.zeros(3))
    return st


def pod_pose(kind, rides, D, shift, st):
    """Body pose of a pod part: its pod-local pose, conjugated by the pod's mirror, then shifted."""
    R, p = st[kind]
    if rides:
        Rab = st["Rab"]
        R, p = Rab @ R, Rab @ (p - mv.AB_C) + mv.AB_C
    return D @ R @ D, D @ p + shift


# ------------------------------------------------------------------------------------------------
# Authoring
# ------------------------------------------------------------------------------------------------
def author(meta, verts, tris, lay):
    F = mv.Film(meta, verts)
    F.tris = tris
    F.theme = "dark"
    F.hidden = set()
    F.up_at = lambda t: (np.eye(3), np.zeros(3))
    POD = lambda pod: [k for k in meta if meta[k].get("pod") == pod]
    base = lambda k: meta[k].get("base", "")
    Q = lambda *n: [k for k in meta if k.startswith(tuple(f"Q_{x}" for x in n)) and "fit" not in meta[k]]
    H = lambda *n: [k for k in meta if k.startswith(tuple(f"Q_{x}" for x in n)) and "fit" in meta[k]]
    # pod hardware fitted in the film: the hub screws (v9.1: into the hub's heat-set inserts)
    hub_screws = {p: [k for k in POD(p) if base(k).startswith("Abad_HW_hub_bolts") and meta[k].get("fit", {}).get("kind") == "screw"]
                  for p in PODS}
    # v9.1 build order: each rear end bracket goes onto its hub first (hub screws driven from its open inner face),
    # then the pod (without that bracket) slides in along the body into the bracket's ears, then the ear screws
    bracket_of = lambda p: [k for k in POD(p) if base(k).startswith("End_Bracket") and not base(k).startswith("End_Bracket_Front")]
    ear_screws = {p: [k for k in POD(p) if base(k).startswith(("FST_Screw_posts_back", "FST_Bolt_posts"))
                      and abs(verts[k][2][0]) < abs(meta[k]["shift"][0]) and "fit" in meta[k]] for p in PODS}
    FIT = ({k for k in meta if k.startswith("Q_") and "fit" in meta[k]} | {k for p in PODS for k in hub_screws[p]}
           | {k for p in PODS for k in ear_screws[p]})
    F.fitted = FIT                                                  # what this film fits itself (check_assembly_quad.py)
    fitinfo = {}
    for k in FIT:
        f = dict(meta[k]["fit"])
        if "pod" in meta[k]:
            f["dir"] = list(np.diag(meta[k]["mirror"]) @ np.array(f["dir"]))
        fitinfo[k] = f
    fixed_of = lambda p, *pre: [k for k in POD(p) if meta[k].get("body_fixed") and base(k).startswith(pre)]
    drum_of = lambda p: [k for k in POD(p) if base(k).startswith(("Abad_Drum", "Abad_REF_Tensioner", "Abad_HW_hub_inserts"))
                         or (base(k).startswith("Abad_HW_hub_bolts") and k not in hub_screws[p])]
    pod_of = lambda p: [k for k in POD(p) if not meta[k].get("body_fixed") and k not in drum_of(p) and k not in hub_screws[p]
                        and k not in bracket_of(p) and k not in ear_screws[p]]
    stand, plate_b = ["Q_Build_Stand"], Q("Body_Plate_Bottom")
    F.base = set(stand + plate_b)
    fr = lambda tag: "F" if tag == "front" else "R"
    BULK = {(kind, s): Q(f"{name}_{s}") for s in "FR" for kind, name in (("end", "End_Frame"), ("inner", "Inner_Bulkhead"),
                                                                          ("centre", "Centre_Bulkhead"))}
    sides = {"F": ("LF", "RF"), "R": ("LR", "RR")}
    motorC = {s: [k for p in sides[s] for k in fixed_of(p, "Abad_Motor", "Abad_moteus", "Abad_HW_motor", "Abad_Small_Drum",
                                                          "Abad_HW_small_drum", "Abad_Bearings_6808")] for s in "FR"}
    innerfix = {s: [k for p in sides[s] for k in fixed_of(p, "Abad_Bearing_tip", "Hip_Posts", "Hip_HW_post")] for s in "FR"}
    endfix = {s: [k for p in sides[s] for k in fixed_of(p, "Hip_Bearing_front")] for s in "FR"}
    cas_parts = Q("Cassette", "REF_Pi5", "REF_pi3hat", "REF_Pack", "REF_power_dist")
    cas_parts = [k for k in cas_parts if not k.startswith(("Q_Cassette_Dock", "Q_REF_Pack"))]   # the battery goes in last
    cas_hw = H("pi_", "hat_", "pd_")
    rides = set()                                                  # (v3: every insert has its own step; the dock's on the bench)
    for k in FIT:
        if k in rides:
            continue
        f = fitinfo[k]
        dist = {"screw": f["len"] + 10.0, "insert": 14.0, "nut": 12.0}.get(f["kind"], 12.0)
        F.offset([k], np.array(f["dir"]) * dist, turns=min(6.0, max(2.0, f["len"] / 3.0)) if f["kind"] == "screw" else 0.0)
    # cassette: built on the bench, carried over the bay, dropped straight down into its dock (offsets in that order)
    bench = np.array([0.0, 560.0, 0.0])
    lift = np.array([0.0, 0.0, 330.0])
    F.offset([k for k in cas_parts if k != "Q_Cassette"], (0, 0, 110))
    F.offset(["Q_Cassette"], (0, 0, 60))
    F.offset(cas_parts + cas_hw, bench)
    F.offset(cas_parts + cas_hw, lift)
    F.reach.update({k: 900.0 for k in cas_parts + cas_hw})
    F.offset(Q("REF_Pack_6S"), (0, 0, 260.0))                     # into the top tray, once the cassette is in
    F.offset(Q("REF_EStop"), (0, 0, 160.0))
    # inserts ride with the part they are pressed into (pressed in while it waits)
    host = {"front": "end", "inner": "inner", "outer": "centre"}
    ins_of = {}
    for k in H("ins_", "nut_"):
        _, _, s_, kind, _ = k[2:].split("_")
        ins_of.setdefault((host[kind], s_), []).append(k)
    for k in H("lap_scr", "lap_nut"):                              # the halves' joints: on the bench, with their bulkhead
        _, _, s_, kind, _ = k[2:].split("_")
        ins_of.setdefault(("inner" if kind == "inner" else "centre", s_), []).append(k)
    lid_ins = H("lid_ins")                                         # LID_PTS: 0, 1 at the rear centre bulkhead, 2, 3 at the front
    ins_of.setdefault(("centre", "R"), []).extend(lid_ins[:2])
    ins_of.setdefault(("centre", "F"), []).extend(lid_ins[2:])
    for kb in list(BULK):
        BULK[kb] = BULK[kb] + ins_of.get(kb, [])
    # body groups
    z = lambda dz: np.array([0.0, 0.0, dz])
    for s in "FR":
        F.offset(BULK[("centre", s)] + motorC[s], z(220))
        F.offset(BULK[("inner", s)] + innerfix[s], z(220))
        F.offset(BULK[("end", s)] + endfix[s], (260.0 if s == "F" else -260.0, 0, 0))
    for p in PODS:
        sx = 1.0 if p[1] == "F" else -1.0
        F.offset(drum_of(p), z(200))
        F.offset(bracket_of(p), (sx * 240.0, 0, 60.0))                               # in along the body, onto the hub's end
        F.offset(pod_of(p), (sx * 420.0, 0, 30.0))                                   # slides in along the body
        F.reach.update({k: 520.0 for k in pod_of(p)})
        # (the hub and ear screws are fitted after their bracket / pod is in: only their own approach offsets)
    F.offset(Q("Body_Plate_Top"), z(200))
    F.offset(Q("Cassette_Dock") + H("dock_ins"), z(120))
    F.offset(Q("Lid"), z(160))
    for k in meta:
        F.reach[k] = max(F.reach.get(k, 0.0), 420.0)
    F.explode = (1000.0, 1001.0)
    F.hide_at = 3.6
    c0 = np.array([0.0, 0.0, -40.0])
    F.cam(0.0, c0, 1650, -50, -24)
    F.cam(3.4, c0, 1600, -25, -20)
    F.cards.append((0.2, 3.3, "Capstan quadruped", "Four v9.1 leg pods, hip ab/ad built into the body  ·  build, walk, jump"))
    F.t = 3.8
    count = iter(range(1, 400))
    S = lambda *a, **kw: F.step(next(count), *a, **kw)

    def view_for(d, az, el=-22.0):
        a_, e_ = math.radians(az), math.radians(el)
        fs = np.array([math.cos(e_) * math.cos(a_), math.cos(e_) * math.sin(a_), math.sin(e_)])
        frm = np.asarray(d, float) / np.linalg.norm(d) - 0.8 * fs
        f = -frm / np.linalg.norm(frm)
        return math.degrees(math.atan2(f[1], f[0])), max(-65.0, min(30.0, math.degrees(math.asin(f[2]))))

    def fit(head, detail, labs, az=-60.0, context=(), hold=0.8, view=None, min_dist=120.0, el=None):
        seq = sorted([k for k in labs if k in FIT], key=lambda k: ({"insert": 0}.get(fitinfo[k]["kind"], 2),
                                                                     math.atan2(verts[k][2][1], verts[k][2][0])))
        if not seq:
            return
        n = len(seq)
        stagger = 0.4 if n <= 4 else 0.2 if n <= 8 else 0.1 if n <= 24 else 0.06
        start, lead = F.t, 0.7
        first_mt = 1.4 if fitinfo[seq[0]]["kind"] == "screw" else 1.0
        t, ends = start + lead, []
        for i, k in enumerate(seq):
            mt = first_mt if i == 0 else 0.8
            F.move([k], t, t + mt)
            ends.append(t + mt)
            t = t + mt + 0.15 if i == 0 else t + stagger
        end = max(ends)
        for k in context:
            if k not in F.base:
                F.appear[k] = min(F.appear.get(k, 1e9), start)
        f0 = fitinfo[seq[0]]
        a_, e_ = view if view is not None else view_for(f0["dir"], az)
        e_ = e_ if el is None else el                             # el: look down into a bay (the laps' faces face each other)
        travel = np.linalg.norm(F.offsets[seq[0]][0][0])
        p0 = verts[seq[0]][2] + F.offset_at(seq[0], ends[0] + 0.01) + np.array(f0["dir"]) * travel * 0.45
        dist = float(np.clip(2.6 * (f0["len"] + travel) + 60.0, min_dist, max(min_dist, 340.0)))
        close = (p0, dist, a_ + 4, e_)
        F.cam(start + lead - 0.5, p0, dist, a_, e_)
        F.cam(ends[0] + 0.25, *close)
        tips = lambda k, tt: [verts[k][2] + F.offset_at(k, tt) + np.array(fitinfo[k]["dir"]) * (fitinfo[k]["len"] * 0.5 + 0.3)]
        F.clear_view(tips(seq[0], start + lead) + [verts[seq[0]][2] + F.offset_at(seq[0], start + lead)] + tips(seq[0], ends[0] + 0.01),
                     close, ends[0], start, end + hold + 0.3, skip=seq)
        if n > 1:
            wide = F.frame(seq, end, a_ + 8, e_, margin=1.3, extra_t=start + lead)
            wide = (wide[0], max(wide[1], 260.0), wide[2], wide[3])
            F.cam(end + hold, *wide)
            F.clear_view([p for k in seq[1:] for p in tips(k, end + 0.01)], wide, end, ends[0], end + hold + 0.3, skip=seq)
        F.highlight.append((seq, start + lead - 0.4, end + 0.3))
        F.captions.append((start + 0.15, end + hold - 0.1, str(next(count)), head, detail))
        F.t = end + hold

    # --- the electronics cassette, on the bench ---
    S("Electronics cassette", "On the bench: one printed rack; power board on the floor, computer shelf, battery tray on top",
      [["Q_Cassette"]], -55, -28, margin=1.6)
    fit("Power board inserts", "4× M2.5 heat-set inserts in the floor's posts", H("pd_ins"), -60)
    S("mjbots power_dist", "On the floor: XT30s out to the four legs low down, the battery lead up to the tray",
      [Q("REF_power_dist")], -55, -26, margin=1.8)
    fit("Power board screws", "4× M2.5 × 6", H("pd_scr"), -60)
    fit("Computer inserts", "4× M2.5 heat-set inserts in the middle shelf's posts", H("pi_ins"), -60)
    S("Raspberry Pi 5", "On the middle shelf's posts, a low heatsink on it", [Q("REF_Pi5")], -55, -24, margin=1.8)
    fit("Standoffs", "4× M2.5 × 12 male-female through the Pi into the inserts", H("pi_so"), -60)
    S("mjbots pi3hat", "CAN-FD: each leg's three moteus on its own bus", [Q("REF_pi3hat")], -55, -24, margin=1.8)
    fit("Hat screws", "4× M2.5 × 6", H("hat_scr"), -60)
    # --- body ---
    for kind, name in (("inner", "Inner_Bulkhead"), ("outer", "Centre_Bulkhead")):     # each joined on its own, on the bench
        for s_, end_ in (("F", "front"), ("R", "rear")):           # one bulkhead at a time, seen from above its bay face
            fit(f"{name.split('_')[0]} bulkhead halves, {end_}", "The two halves lapped: 3× M3 × 12 countersunk, flush on the bay face",
                H(f"lap_scr_{s_}_{kind}"), -60, context=Q(name), min_dist=380.0, el=-50.0)
            fit(f"{name.split('_')[0]} lap nuts, {end_}", "3× M3 nylon-insert nuts on the outboard face",
                H(f"lap_nut_{s_}_{kind}"), 120, context=Q(name), min_dist=380.0, el=-45.0)
    fit("Edge inserts", "M3 heat-set inserts in every bulkhead's top and bottom edge bosses (and the lid's four in the centre ones)",
        H("ins_", "lid_ins"), -60, context=Q("End_Frame", "Inner_Bulkhead", "Centre_Bulkhead"), min_dist=200.0)
    fit("Dock inserts", "4× M3 heat-set inserts into the dock's floor from below, on the bench (it waits above the plate)",
        H("dock_ins"), -60, view=(-60.0, 35.0), context=Q("Cassette_Dock"))
    S("Cassette dock", "Onto the bottom plate, centred in the bay", [Q("Cassette_Dock") + H("dock_ins")],
      -60, -30, margin=1.6, context=plate_b)
    fit("Dock screws", "4× M3 countersunk from under the plate into the dock's inserts", H("dock_scr"), -60, view=(-60.0, 30.0))
    S("Centre bulkheads", "Each carries both its pods' ab/ad motors C (moteus C on the bay side) and the hubs' outer 6808s",
      [BULK[("centre", "F")] + motorC["F"], BULK[("centre", "R")] + motorC["R"]], -55, -22, margin=1.1, stagger=0.6)
    fit("Centre bulkhead screws", "M3 × 8 countersunk from under the plate into the edge inserts", H("scr_bot_F_outer", "scr_bot_R_outer"), -60,
        view=(-60.0, 30.0))
    S("Ab/ad drums", "Each hub into its outer 6808; sector toward its pod; tension pegs and the hub's 6 inserts already in",
      [drum_of(p) for p in PODS], -55, -22, margin=1.0, stagger=0.4)
    st = F.t                                                       # the ab/ad ropes, while the bay is open from above
    F.ab_rope_t = st + 0.6
    c0 = F.frame([k for p in PODS for k in drum_of(p)], st + 0.5, -55.0, -30.0, 1.0)
    F.cam(st + 0.3, *c0)
    F.cam(st + 5.6, c0[0], c0[1], c0[2] + 10.0, c0[3])
    F.highlight.append(([k for p in PODS for k in drum_of(p) if base(k).startswith("Abad_REF_Tensioner")], st + 0.5, st + 5.4))
    F.captions.append((st + 0.2, st + 5.7, str(next(count)), "Ab/ad ropes",
                       "Each loop round motor C's small drum and onto the sector's two pegs, wound tight from above (3 mm key)"))
    F.t = st + 5.9
    S("Inner bulkheads", "Over the hubs (inner 6808s) and motor C's tip; the ab/ad posts to the centre bulkheads",
      [BULK[("inner", "F")] + innerfix["F"], BULK[("inner", "R")] + innerfix["R"]], -55, -22, margin=1.0, stagger=0.6)
    fit("Inner bulkhead screws", "M3 × 8 countersunk from under the plate into the edge inserts", H("scr_bot_F_inner", "scr_bot_R_inner"), -60,
        view=(-60.0, 30.0))
    S("Rear end brackets onto the hubs", "Each web against its hub's end; the bracket is alone, so its inner face is open",
      [bracket_of(p) for p in PODS], -60, -20, margin=1.0, stagger=0.5)
    for p in PODS:
        fit(f"{p} hub screws", "8× M4 × 16 through the bracket's web into the hub's inserts, from the open side",
            hub_screws[p], view=((35.0 if p[0] == "L" else -35.0) + (0.0 if p[1] == "F" else 180.0), -35.0), min_dist=480.0)
    for pair, head, az in ((("LF", "LR"), "Right-side leg pods", -70), (("RF", "RR"), "Left-side leg pods (mirror images)", 110)):
        S(head, "Each pod slides in along the body: its corner posts into the bracket's ears, the trunnion pointing out",
          [pod_of(p) for p in pair], az, -18, margin=1.0, stagger=0.8, move_time=1.8)
    for p in PODS:
        fit(f"{p} ear screws", "M4 × 25 through the bracket's ears into the posts' inserts, front and back", ear_screws[p],
            view=(-90.0 if p[0] == "L" else 90.0, 10.0), min_dist=300.0)
    S("End frames", "Stadium frames over both trunnions (6806s pressed in); the pods now hang on two bearings each",
      [BULK[("end", "F")] + endfix["F"], BULK[("end", "R")] + endfix["R"]], -40, -18, margin=1.0, stagger=0.6)
    fit("End frame screws", "M3 × 8 countersunk from under the plate into the edge inserts", H("scr_bot_F_front", "scr_bot_R_front"), -60,
        view=(-60.0, 30.0))
    S("Top plate", "3 mm laser-cut carbon or 6061; the bulkheads' tabs key into its slots; the cassette's opening in the middle",
      [Q("Body_Plate_Top")], -50, -40, margin=1.0)
    fit("Top plate screws", "18× M3 × 8 countersunk into the edge inserts", H("scr_top"), -50, view=(-50.0, -60.0), min_dist=300.0)
    S("E-stop", "22 mm latching e-stop into the top plate's hole, ring nut from below; fixed to the body, not the lid",
      [Q("REF_EStop")], -40, -35, margin=2.0)
    S("Cassette to the bay", "Carried over the opening by the tray's end stops (its handles)", [cas_parts + cas_hw], -40, -24,
      margin=1.2, move_time=1.6)
    S("Cassette in", "Down into the dock between the moteus C boards; leg harness and e-stop plugged in, loops in the middle",
      [cas_parts + cas_hw], -40, -24, margin=1.3, move_time=1.8)
    S("Battery", "Into the top tray, strapped; plugged into the power board's lead. To swap it: lid off, unplug, unstrap, lift",
      [Q("REF_Pack_6S")], -40, -30, margin=1.6)
    S("Lid", "Over the opening; two ribs under it hold the cassette down; vents over the pack's sides", [Q("Lid")], -40, -30,
      margin=1.4)
    fit("Lid thumb screws", "4× M3 thumb screws into the centre bulkheads' inserts: no tools to swap the battery",
        H("lid_scr"), -40, view=(-40.0, -50.0), min_dist=280.0)
    return F, F.t, POD


# ------------------------------------------------------------------------------------------------
# Motion: stand up, trot, jump
# ------------------------------------------------------------------------------------------------
class Motion:
    def __init__(self, meta, t0):
        self.core = {}
        for p in PODS:
            k = f"{p}__Motor_Plate"
            D = np.diag(meta[k]["mirror"])
            self.core[p] = (D, np.array(meta[k]["shift"]))
        self.hip = {p: self.core[p][0] @ np.array([mv.OUT[0], 0.0, mv.OUT[1]]) + self.core[p][1] for p in PODS}
        self.foot_y = {p: (self.core[p][0] @ np.array([0.0, 123.5, 0.0]) + self.core[p][1])[1] for p in PODS}
        self.r_fold, self.r_walk = 215.0 * mv.L4_K, float(mv.L4_R0)
        self.h_fold = FOOT_R + self.r_fold - HIP_Z
        self.h_walk = FOOT_R + self.r_walk - HIP_Z
        self.T_up, self.T_walk, self.period, self.stride, self.lift = 3.2, 11.0, 0.72, 110.0 * mv.L4_K, 42.0 * mv.L4_K
        self.dir = -1.0                # walks toward -X: the knees (which point to +X) trail, as on a dog
        self.track = mv.hop_track(mv.LEGS["4bar"], slow=0.45)
        self.tt = np.cumsum([0.0] + [x[0] for x in self.track])[:-1]
        self.t_up = t0 + 0.4
        self.t_walk = self.t_up + self.T_up + 0.6
        self.t_jump = self.t_walk + self.T_walk + 0.8
        self.T_jump = float(self.tt[-1])
        self.t_jump2 = self.t_jump + self.T_jump + 0.3
        self.t_end = self.t_jump2 + self.T_jump + 1.2
        ts = np.linspace(0, self.T_walk, 2000)
        v = np.array([self.speed(u) for u in ts])
        self.xs, self.ts = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) / 2 * np.diff(ts))]), ts

    def env(self, u):
        return max(0.0, min(1.0, u / self.period, (self.T_walk - u) / self.period))

    def speed(self, u):
        return self.dir * self.env(u) * self.stride / (self.period / 2)

    def body(self, t):
        if t < self.t_up:
            return 0.0, self.h_fold
        if t < self.t_walk:
            return 0.0, self.h_fold + (self.h_walk - self.h_fold) * mv.ease(min(1.0, (t - self.t_up) / self.T_up))
        if t < self.t_jump:
            u = t - self.t_walk
            return float(np.interp(min(u, self.T_walk), self.ts, self.xs)), \
                self.h_walk + 4.0 * math.sin(4 * math.pi * u / self.period) * self.env(u)
        h = self.h_walk
        for tj in (self.t_jump, self.t_jump2):
            if tj <= t < tj + self.T_jump:
                h = float(np.interp(t - tj, self.tt, [x[3] for x in self.track])) - HIP_Z
        return float(self.xs[-1]), h

    def feet(self, t):
        bx, bz = self.body(t)
        out = {}
        for p in PODS:
            hx, hz = self.hip[p][0], self.hip[p][2]
            fz = FOOT_R - bz
            if t < self.t_up:
                out[p] = np.array([hx, self.foot_y[p], hz - self.r_fold])
                continue
            if self.t_walk <= t < self.t_jump:
                u = t - self.t_walk
                ph = (u / self.period + (0.0 if p in ("LF", "RR") else 0.5)) % 1.0
                s = self.stride * self.env(u)
                if ph < 0.5:
                    dx, dz = self.dir * s * (0.5 - ph / 0.5), 0.0
                else:
                    w = (ph - 0.5) / 0.5
                    dx, dz = self.dir * (-s / 2 + s * mv.ease(w)), self.lift * math.sin(math.pi * w) * min(1.0, s / 40.0)
                out[p] = np.array([hx + dx, self.foot_y[p], fz + dz])
                continue
            r = None
            for tj in (self.t_jump, self.t_jump2):
                if tj <= t < tj + self.T_jump:
                    r = float(np.interp(t - tj, self.tt, [x[1] for x in self.track]))
            out[p] = np.array([hx, self.foot_y[p], hz - r]) if r is not None else np.array([hx, self.foot_y[p], fz])
        return out

    def angles(self, p, foot):
        D, t = self.core[p]
        loc = D @ (np.asarray(foot) - t)
        dx, dz = loc[0] - mv.OUT[0], loc[2] - mv.OUT[1]
        r = min(math.hypot(dx, dz), mv.L4_L1 + mv.L4_L2 - 2.0)
        return mv.leg4_ik(r, math.degrees(math.atan2(dz, dx)))


# ------------------------------------------------------------------------------------------------
# Render
# ------------------------------------------------------------------------------------------------
CAM_MOTION = []


def camera(F, M, t):
    bx, bz = M.body(t)
    if t < M.t_up - 0.2:
        lookat, dist, az, el = F.camera(t)
        return np.asarray(lookat, float) + np.array([bx, 0.0, bz]), dist, az, el
    keys = CAM_MOTION
    k = keys[0] if t <= keys[0][0] else keys[-1]
    for (ta, la, da, aa, ea), (tb, lb, db, ab, eb) in zip(keys, keys[1:]):
        if ta <= t <= tb:
            u = mv.ease((t - ta) / (tb - ta))
            k = (t, la + (lb - la) * u, da + (db - da) * u, aa + (ab - aa) * u, ea + (eb - ea) * u)
            break
    return np.array([bx, 0.0, 0.0]) + k[1], k[2], k[3], k[4]


class Poser:
    """Poses the CAD scene at film time t: assembly offsets, fades and highlights, and the legs' motion."""

    def __init__(self, F, M, meta, model, data):
        self.F, self.M, self.meta, self.model, self.data = F, M, meta, model, data
        bid = {k: model.body(k).id for k in meta}
        self.mocap = {k: model.body_mocapid[i] for k, i in bid.items()}
        self.geom_of = {k: model.body_geomadr[i] for k, i in bid.items()}
        self.mat_of = {k: model.geom_matid[gg] for k, gg in self.geom_of.items()}
        self.base_rgba = {mm: model.mat_rgba[mm].copy() for mm in set(self.mat_of.values())}
        self.pods = {k: (classify(meta[k]), np.diag(meta[k]["mirror"]), np.array(meta[k]["shift"])) for k in meta if "pod" in meta[k]}

    def __call__(self, t):
        """Set every part's pose at t; returns (body offset pb, per-pod leg states)."""
        F, M, meta, model, data = self.F, self.M, self.meta, self.model, self.data
        quat = np.zeros(4)
        bx, bz = M.body(t)
        pb = np.array([bx, 0.0, bz])
        feet = M.feet(t)
        states = {p: leg_state(*M.angles(p, feet[p]), 0.0) for p in PODS}
        mat_a, mat_g = {}, {}
        hl_now = [(set(labs), g0, g1) for labs, g0, g1 in F.highlight if g0 <= t <= g1]
        for k in meta:
            R0, p0 = F.pose_at(k, t)                        # assembly offsets (and screws turning in)
            if k in self.pods:
                (kind, rides), D, sh = self.pods[k]
                R, p = pod_pose(kind, rides, D, sh, states[meta[k]["pod"]])
                R, p = R0 @ R, R0 @ p + p0
            else:
                R, p = R0, p0
            if k == "Q_Build_Stand" and t >= M.t_up:      # the bench block stays put, then drops away
                drop = 0.0 if t < M.t_up + 0.8 else -700.0 * mv.ease(min(1.0, (t - M.t_up - 0.8) / 1.6))
                p = p + np.array([0.0, 0.0, drop - (bz - M.h_fold)])
            data.mocap_pos[self.mocap[k]] = p + pb
            mujoco.mju_mat2Quat(quat, R.flatten())
            data.mocap_quat[self.mocap[k]] = quat
            a = F.alpha(k, t)
            model.geom_group[self.geom_of[k]] = 5 if a <= 0.01 else 0
            mm = self.mat_of[k]
            glow = 0.0
            for labs, g0, g1 in hl_now:
                if k in labs:
                    glow = max(glow, 0.22 * mv.window(t, g0, g1, 0.4))
            if a > 0.01:
                mat_a[mm] = max(mat_a.get(mm, 0.0), a)
            mat_g[mm] = max(mat_g.get(mm, 0.0), glow)
        for mm, rgba0 in self.base_rgba.items():
            rgba = rgba0.copy()
            rgba[3] = mat_a.get(mm, 1.0)
            model.mat_rgba[mm] = rgba
            model.mat_emission[mm] = mat_g.get(mm, 0.0)
        mujoco.mj_forward(model, data)
        return pb, states

    def ropes(self, scn, t, pb, states):
        F, meta, pods = self.F, self.meta, self.pods
        for p in PODS:                                       # capstan ropes, per pod
            st = states[p]
            kc, kd = f"{p}__Motor_Plate", f"{p}__Abad_Drum"
            if F.alpha(kc, t) > 0.5:
                D, sh = pods[kc][1], pods[kc][2]
                off = F.offset_at(kc, t)
                for stage, motor, y_mid in (("A", mv.MA, mv.A_MID), ("B", mv.MB, mv.B_MID)):
                    pts = mv.rope_path(motor, y_mid, st["th"][stage])
                    pts = (st["Rab"] @ (pts - mv.AB_C).T).T + mv.AB_C
                    mv.draw_rope(scn, pts @ D.T + sh + off + pb, 1.0, 1.3, False)
            if kd in meta and F.alpha(kd, t) > 0.5 and t >= getattr(F, "ab_rope_t", 0.0):
                D, sh = pods[kd][1], pods[kd][2]
                pts = mv.ab_map(mv.rope_path(mv.MB, mv.AB_MID, -st["ab"], ab=True))
                mv.draw_rope(scn, pts @ D.T + sh + F.offset_at(kd, t) + pb, 1.0, 1.3, False)


def render(F, M, meta, model, out, preview_at=None):
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, mv.H, mv.W, max_geom=20000)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 1, 1, 0]
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    poser = Poser(F, M, meta, model, data)
    dur = M.t_end + 3.0
    F.duration = dur
    frames = [int(x * mv.FPS) for x in preview_at] if preview_at else range(int(dur * mv.FPS))
    procs = []
    if not preview_at:
        for path in (out, out.with_name(out.stem + "_notext.mp4")):
            procs.append(subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                           "-s", f"{mv.W}x{mv.H}", "-r", str(mv.FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                                           "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE))
    title = "Capstan quadruped  ·  4× v9.1 leg pod, ab/ad in the body  ·  build, walk, jump"
    for fi in frames:
        t = fi / mv.FPS
        pb, states = poser(t)
        lookat, dist, az, el = camera(F, M, t)
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, dist, az, el
        renderer.update_scene(data, camera=cam, scene_option=opt)
        poser.ropes(renderer.scene, t, pb, states)
        img = Image.fromarray(renderer.render())
        raw = img
        img = mv.overlay(img, F, t, (np.array(lookat), dist, az, el), title, mv.THEMES["dark"])
        if preview_at:
            img.save(out.parent / f"{out.stem}_{fi:05d}.jpg", quality=85)
        else:
            procs[0].stdin.write(img.tobytes())
            procs[1].stdin.write(raw.convert("RGB").tobytes())
        if fi % (mv.FPS * 10) == 0:
            print(f"  {out.name}: {t:5.1f}/{dur:.1f} s", flush=True)
    for pr in procs:
        pr.stdin.close()
        pr.wait()
    renderer.close()


def setup(meta, verts, tris, lay):
    """The whole film's timeline: the build (author), then stand up, trot, jump; with its camera and captions."""
    F, t_built, POD = author(meta, verts, tris, lay)
    M = Motion(meta, t_built)
    z0 = M.h_walk - 120.0
    CAM_MOTION[:] = [
        (M.t_up - 0.2, np.array([0.0, 0.0, M.h_fold - 80.0]), 1700.0, -50.0, -18.0),
        (M.t_walk, np.array([0.0, 0.0, z0]), 1750.0, -70.0, -10.0),
        (M.t_walk + M.T_walk * 0.5, np.array([60.0 * M.dir, 0.0, z0]), 1650.0, -100.0, -8.0),
        (M.t_jump, np.array([0.0, 0.0, z0]), 1800.0, -125.0, -10.0),
        (M.t_end, np.array([0.0, 0.0, z0 + 40.0]), 1850.0, -150.0, -14.0),
        (M.t_end + 3.0, np.array([0.0, 0.0, z0 + 40.0]), 1900.0, -165.0, -16.0),
    ]
    F.captions.append((M.t_up + 0.1, M.t_walk - 0.2, "", "Stand up", "The bench block drops away as the legs take the weight"))
    F.captions.append((M.t_walk + 0.2, M.t_jump - 0.3, "", "Trot", "Diagonal pairs: front-left with rear-right, front-right with rear-left"))
    F.captions.append((M.t_jump + 0.1, M.t_end - 0.2, "", "Jump (shown at 0.45× speed)", "All four legs: crouch, push, tuck, land"))
    F.cards.append((M.t_end + 0.2, M.t_end + 2.9, "Capstan quadruped", "CAD animation  ·  kinematics only, not a dynamics simulation"))
    return F, M, t_built


def main():
    qdir = Path(sys.argv[1]).resolve()
    at = [float(x) for a in sys.argv if a.startswith("--at=") for x in a[5:].split(",")]
    q = json.loads((qdir / "quad.json").read_text())
    meta, lay = q["meta"], q["layout"]
    model = build_model(meta)
    verts, tris = mesh_data(model, meta)
    F, M, t_built = setup(meta, verts, tris, lay)
    out = ROOT / "video" / "quadruped.mp4"
    if at:
        out = qdir.parent / "preview" / "quadruped.mp4"
        out.parent.mkdir(exist_ok=True)
    print(f"quadruped film: build {t_built:.1f} s, total {M.t_end + 3.0:.1f} s, {len(meta)} bodies", flush=True)
    render(F, M, meta, model, out, preview_at=at or None)


if __name__ == "__main__":
    main()
