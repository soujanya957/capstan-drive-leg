"""MuJoCo-rendered assembly film for the v3 coaxial capstan.

Every placed part (printed parts, bearings, screws, inserts, nuts, motor rotor/stator) is a
mocap body whose mesh is in world coordinates, so the film is driven purely kinematically:
parts slide along their insertion paths, the camera frames each step, ropes follow the
crossed-belt geometry, and the finished drive is run through its 8:1 stages at the end.

Outputs  video/assembly.mp4           (intro, explode, step-by-step assembly, drivetrain)
         video/assembly_detailed.mp4  (every part and fastener, one action per step, screws thread in)
         video/exploded.mp4           (labelled exploded view, assembled -> exploded -> assembled)

Run:  python export_meshes.py <mesh_dir>                         (build123d env)
      python make_videos.py <mesh_dir> [assembly|detailed|exploded] [--preview] [--at=t1,t2]
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
import mujoco
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "video"
W, H = (int(v) for v in os.environ.get("CAPSTAN_VIDEO_SIZE", "1920x1080").split("x"))
FPS = int(os.environ.get("CAPSTAN_VIDEO_FPS", "30"))
OW, OH = 1920, 1080                    # the text overlay is laid out at 1080p and scaled to W x H
FOVY = 32.0
APPROACH = 150.0                       # max insertion travel shown during the step-by-step assembly
# The CAD is modelled hanging (motors above the output); the rig stands upright on the desk, so the
# world is the model turned 180 deg about Y with the base underside at Z 0 (generate.UPRIGHT).
UP_R = np.diag([-1.0, 1.0, -1.0])
UP_T = np.array([0.0, 0.0, 55.0 + 8.0])


def to_world(pts):
    return np.atleast_2d(pts) @ UP_R.T + UP_T

# ---------------------------------------------------------------------------------------------
# Geometry (mirrors generate.py; kept here so this script does not need build123d)
# ---------------------------------------------------------------------------------------------
MOTOR_PITCH, CENTER_DIST = 93.5, 84.0
MA, MB = (-MOTOR_PITCH / 2, 0.0), (MOTOR_PITCH / 2, 0.0)
OUT = (0.0, -math.sqrt(CENTER_DIST ** 2 - (MOTOR_PITCH / 2) ** 2))
SMALL_R, BIG_R, RATIO = 70.0 / 9.0, 70.0, 9.0       # v9: 9:1 (generate.SMALL_PITCH_R)
TANGENT_DEG = math.degrees(math.acos((BIG_R + SMALL_R) / CENTER_DIST))
TRAVEL_HALF, GROOVE_PITCH = 90.0, 1.9  # v8 (generate.TRAVEL_HALF, GROOVE_PITCH)
ROPE_HOLE_DEG = TRAVEL_HALF + TANGENT_DEG + 2.0
A_MID, B_MID = 52.25, 76.75           # v8 slabs 41-63.5 and 65.5-88
AB_MID = 79.0                          # the ab/ad capstan keeps v6.2's drum-B stage (70-88)
ROPE_HALF = 5.2                        # strands leave the 1.9 mm helix 2.75 turns either side of the anchor
DIR_A = math.degrees(math.atan2(MA[1] - OUT[1], MA[0] - OUT[0]))
DIR_B = math.degrees(math.atan2(MB[1] - OUT[1], MB[0] - OUT[0]))
ARM_DIR = {"A": DIR_A - 210.0, "B": DIR_B - 90.0}     # XZ angle of each test arm at rest
ARM_Y = {"A": 132.0, "B": 115.0}
# In-line rope tensioner (mirrors generate.ts_loc): rope, slider and M3 screw on one straight line
RIM_R, SECTOR_HALF = 69.25, ROPE_HOLE_DEG + 21.4
TS_D = 50.0
TS_THC = ROPE_HOLE_DEG + math.degrees(math.acos(TS_D / RIM_R))
TS_END = TS_D * math.tan(math.radians(SECTOR_HALF - TS_THC))
TS_RIM = -math.sqrt(RIM_R ** 2 - TS_D ** 2)
TS_SPOT, TS_WALL, TS_GAP, TS_TRAVEL, TS_ZR, TS_ZS = 1.2, 5.0, 0.3, 4.0, 2.75, 6.5
PEG_X, PEG_Z = 11.0, 4.5                                     # v6 winding peg (generate.PEG_*): rope crossing
DRUM_Y = {"A": (41.0, 63.5), "B": (65.5, 88.0)}
# the ab/ad capstan's sector (v6.2's): its own rope hole, end and stage
AB_GEO = dict(thc=56.6 + math.degrees(math.acos(TS_D / RIM_R)), y=(70.0, 88.0))
AB_GEO["end"] = TS_D * math.tan(math.radians(78.0 - AB_GEO["thc"]))


def ts_frame(stage, s, ab=False):
    """(origin, x_dir, z_dir) in model coordinates of the tensioner at sector end s (generate's sign):
    x runs from the end face toward the rim, z is depth from the pocket's open face."""
    y0, y1 = AB_GEO["y"] if ab else DRUM_Y[stage]
    thc, end = (AB_GEO["thc"], AB_GEO["end"]) if ab else (TS_THC, TS_END)
    phi_c = math.radians(-(DIR_A if stage == "A" else DIR_B))
    th = math.radians(s * thc)
    n = np.array([math.cos(th), math.sin(th)])
    u = s * np.array([-math.sin(th), math.cos(th)])
    rot = np.array([[math.cos(phi_c), -math.sin(phi_c)], [math.sin(phi_c), math.cos(phi_c)]])
    o, xd = rot @ (TS_D * n + end * u), rot @ -u
    to_m = lambda v, z: np.array([v[0], z, -v[1]])                      # generate.to_y: (x, y, z) -> (x, z, -y)
    origin = to_m(o, (y1 if s > 0 else y0)) + np.array([OUT[0], 0.0, OUT[1]])
    return origin, to_m(xd, 0.0), np.array([0.0, -1.0 if s > 0 else 1.0, 0.0])


def ts_point(stage, s, x, depth, ab=False):
    o, xd, zd = ts_frame(stage, s, ab)
    return o + x * xd + depth * zd


def ts_of(label):
    """(stage, generate's end sign) of a tensioner label."""
    return label[len("REF_Tensioner_")], (1 if "_plus_" in label else -1)


SLIDER = {("A", -1): "Rope_Tensioner_Slider_1", ("A", 1): "Rope_Tensioner_Slider_2",
          ("B", -1): "Rope_Tensioner_Slider_3", ("B", 1): "Rope_Tensioner_Slider_4"}

PALETTE = {  # rgb, specular, shininess, reflectance
    "gray": ((0.70, 0.71, 0.72), 0.25, 0.3, 0.0),
    "dimgray": ((0.38, 0.39, 0.41), 0.3, 0.4, 0.0),
    "silver": ((0.80, 0.81, 0.83), 0.6, 0.7, 0.05),
    "red": ((0.80, 0.22, 0.17), 0.35, 0.45, 0.0),
    "teal": ((0.10, 0.56, 0.58), 0.35, 0.45, 0.0),           # v9.1: drum A
    "amber": ((0.93, 0.52, 0.12), 0.35, 0.45, 0.0),          # v9.1: drum B
    "violet": ((0.47, 0.32, 0.70), 0.35, 0.45, 0.0),         # v9.1: the ab/ad drum
    "black": ((0.11, 0.11, 0.12), 0.45, 0.5, 0.0),                  # v4: Stanley's black plates
    "white": ((0.93, 0.93, 0.90), 0.3, 0.4, 0.0),
    "yellow": ((0.93, 0.74, 0.18), 0.35, 0.45, 0.0),
    "motor": ((0.74, 0.55, 0.20), 0.7, 0.8, 0.08),
    "motor_dark": ((0.30, 0.30, 0.32), 0.5, 0.6, 0.03),
    "pcb": ((0.15, 0.45, 0.30), 0.3, 0.4, 0.0),
    "steel": ((0.62, 0.66, 0.72), 0.9, 0.9, 0.12),
    "screw": ((0.80, 0.81, 0.84), 0.9, 0.9, 0.08),         # zinc-plated: reads against the black and red prints
    "brass": ((0.86, 0.66, 0.28), 0.9, 0.8, 0.1),
    "magnet": ((0.72, 0.73, 0.76), 0.9, 0.9, 0.15),
    "desk": ((0.62, 0.47, 0.32), 0.15, 0.2, 0.0),
}
ROPE_RGBA = (0.10, 0.55, 1.0, 1.0)
BEAD_RGBA = (1.0, 1.0, 1.0, 1.0)

# parts that turn with each stage
SPIN = {
    "A": ("Big_Drum_A", "Rope_Tensioner_Slider_1", "Rope_Tensioner_Slider_2",
          "REF_Tensioner_A", "Output_Flange_A", "Test_Arm_A", "FST_Insert_A", "FST_Screw_A", "A_Shaft_Cap", "Leg_Thigh_A", "Leg4_Crank"),
    "B": ("Big_Drum_B", "Rope_Tensioner_Slider_3", "Rope_Tensioner_Slider_4", "REF_Tensioner_B",
          "Output_Flange_B", "Test_Arm_B", "FST_Insert_B", "FST_Screw_B", "FST_Nut_B", "Leg_Thigh_B", "Leg4_Thigh"),
    "mA": ("Small_Drum_A", "FST_Rotor_A", "REF_Motor_A_rotor", "REF_Motor_A_magnet"),
    "mB": ("Small_Drum_B", "FST_Rotor_B", "REF_Motor_B_rotor", "REF_Motor_B_magnet"),
}


def labels_of(meta, *names):
    out = []
    for n in names:
        out += [k for k, v in meta.items() if k == n or v["part"] == n or k.startswith(n)]
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------------------------
# MuJoCo model
# ---------------------------------------------------------------------------------------------
THEMES = {
    "light": {"sky": ("0.97 0.97 0.96", "0.80 0.83 0.87"), "floor": None, "ink": (40, 46, 56)},
    "dark": {"sky": ("0.35 0.5 0.7", "0.02 0.03 0.06"), "floor": -0.5, "ink": (226, 232, 240)},
}


def build_model(mesh_dir, meta, theme="light"):
    assets, bodies, materials = [], [], []
    for label, info in meta.items():
        rgb, spec, shin, refl = PALETTE[info["color"]]
        materials.append(f'<material name="m_{label}" rgba="{rgb[0]} {rgb[1]} {rgb[2]} 1" specular="{spec}" '
                         f'shininess="{shin}" reflectance="{refl}"/>')
        assets.append(f'<mesh name="{label}" file="{mesh_dir}/{info.get("file", label)}.stl" inertia="shell"/>')
        bodies.append(f'<body name="{label}" mocap="true"><geom type="mesh" mesh="{label}" '
                      f'material="m_{label}" contype="0" conaffinity="0" mass="0.01"/></body>')
    rgb = PALETTE["desk"][0]
    th = THEMES[theme]
    floor = ""
    if th["floor"] is not None:
        floor = (f'<geom name="floor" type="plane" size="3000 3000 1" pos="0 200 {th["floor"]}" material="floor" '
                 f'contype="0" conaffinity="0"/>')
    xml = f"""
<mujoco model="coaxial_capstan_v3">
  <compiler meshdir="/" angle="radian"/>
  <statistic extent="450" center="0 60 -60"/>
  <visual>
    <global offwidth="{W}" offheight="{H}" fovy="{FOVY}"/>
    <quality shadowsize="8192" offsamples="8"/>
    <map znear="0.002" zfar="40" shadowclip="1.2" shadowscale="0.5" haze="0.15"/>
    <headlight ambient="0.32 0.32 0.33" diffuse="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="{th["sky"][0]}" rgb2="{th["sky"][1]}" width="64" height="1024"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.18 0.2 0.23" rgb2="0.12 0.13 0.15" mark="edge" markrgb="0.3 0.33 0.36" width="512" height="512"/>
    <material name="floor" texture="grid" texrepeat="30 30" reflectance="0.2"/>
    <material name="m_desk" rgba="{rgb[0]} {rgb[1]} {rgb[2]} 1" specular="0.15" shininess="0.2"/>
    {''.join(materials)}
    {''.join(assets)}
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 1" dir="-0.35 -0.45 -0.82" diffuse="0.55 0.55 0.53" specular="0.25 0.25 0.25" castshadow="true"/>
    <light directional="true" pos="0 0 1" dir="0.6 0.3 -0.4" diffuse="0.18 0.19 0.22" specular="0 0 0" castshadow="false"/>
    <light directional="true" pos="0 0 1" dir="0.0 -0.8 0.3" diffuse="0.10 0.10 0.11" specular="0 0 0" castshadow="false"/>
    {floor}
    <body name="desk" mocap="true">
      <geom type="box" size="420 420 18" pos="0 -40 -18.6" material="m_desk" contype="0" conaffinity="0" mass="1"/>
    </body>
    {''.join(bodies)}
  </worldbody>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


# ---------------------------------------------------------------------------------------------
# Rope geometry: crossed (figure-8) loop between a big drum and its small drum
# ---------------------------------------------------------------------------------------------
def rope_path(motor, y_mid, theta, shift=None, ab=False):
    """Polyline from the knot in one tensioner slider, out of the rim, across the crossed span, round
    the small-drum helix, to the other slider. shift[s] moves strand s's knot (slider not yet tight)."""
    cx, cz = OUT
    mx, mz = motor
    stage = "A" if mx < 0 else "B"
    phi0 = math.atan2(mz - cz, mx - cx)
    tdeg = math.radians(TANGENT_DEG)
    c, si = math.cos(theta), math.sin(theta)
    turn = lambda p: np.array([cx + c * (p[0] - cx) - si * (p[2] - cz), p[1], cz + si * (p[0] - cx) + c * (p[2] - cz)])
    strands = []
    for s in (+1, -1):                                       # strand s ends in generate's tensioner -s
        walk = theta * RATIO / (2 * math.pi) * GROOVE_PITCH                          # v8: the wrap walks along the helix
        y = y_mid - s * ROPE_HALF + walk
        n_ang = phi0 + s * tdeg
        knot = ts_point(stage, -s, PEG_X, PEG_Z, ab) + (shift[s] if shift else 0.0)         # on the peg
        exit_ = turn(ts_point(stage, -s, (AB_GEO["end"] if ab else TS_END) - TS_RIM, TS_ZR, ab))
        anchor = math.atan2(exit_[2] - cz, exit_[0] - cx)
        anchor = n_ang + (anchor - n_ang + math.pi) % (2 * math.pi) - math.pi     # short way round, never across ±180°
        pts = [tuple(turn(knot)), tuple(exit_)]                                       # straight through the drum
        for a in np.linspace(anchor, n_ang, 40):                                      # wrap on the big drum
            pts.append((cx + BIG_R * math.cos(a), y, cz + BIG_R * math.sin(a)))
        pts.append((mx - SMALL_R * math.cos(n_ang), y, mz - SMALL_R * math.sin(n_ang)))  # crossed free span
        strands.append(np.array(pts))
    q1, q2 = strands[0][-1], strands[1][-1]
    a1 = math.atan2(q1[2] - mz, q1[0] - mx)
    a2 = math.atan2(q2[2] - mz, q2[0] - mx)
    span = strands[0][-1] - strands[0][-2]
    sense = np.sign((q1[0] - mx) * span[2] - (q1[2] - mz) * span[0])               # winding direction
    total = (a2 - a1) * sense % (2 * math.pi) + 2 * math.pi * 4
    k = np.linspace(0, 1, 260)
    helix = np.stack([mx + SMALL_R * np.cos(a1 + sense * total * k), q1[1] + (q2[1] - q1[1]) * k,
                      mz + SMALL_R * np.sin(a1 + sense * total * k)], axis=1)
    return np.vstack([strands[0], helix[1:-1], strands[1][::-1]])


def add_geom(scn, kind, size, pos, rgba):
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, kind, np.array(size, float), np.array(pos, float), np.eye(3).flatten(),
                        np.array(rgba, dtype=np.float32))
    scn.ngeom += 1
    return g


def draw_rope(scn, pts, frac, width, beads=False):
    if frac <= 0:
        return
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    stop = frac * cum[-1]
    for i in range(len(pts) - 1):
        if cum[i] >= stop:
            break
        p0, p1 = pts[i], pts[i + 1]
        if cum[i + 1] > stop:
            p1 = p0 + (p1 - p0) * (stop - cum[i]) / seg[i]
        g = add_geom(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, (0, 0, 0), (0, 0, 0), ROPE_RGBA)
        if g is None:
            return
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, p0, p1)
    if beads:  # material points at fixed distance from the anchor, so they travel with the rope
        for u in np.arange(6.0, min(stop, cum[-1]), 9.0):
            i = min(np.searchsorted(cum, u) - 1, len(seg) - 1)
            p = pts[i] + (pts[i + 1] - pts[i]) * (u - cum[i]) / seg[i]
            add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (width * 1.25, 0, 0), p, BEAD_RGBA)


# ---------------------------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------------------------
def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (6 * u - 15) + 10)


def axis_rot(axis, angle):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def ray_hits(tri, o, d, s_max):
    """Does the segment o + s d (0 < s < s_max) cross any of the triangles (N, 3, 3)? (Moller-Trumbore)"""
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    pv = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, pv)
    ok = np.abs(det) > 1e-9
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tv = o - tri[:, 0]
    u = np.einsum("ij,ij->i", tv, pv) * inv
    qv = np.cross(tv, e1)
    v = (qv @ d) * inv
    s = np.einsum("ij,ij->i", e2, qv) * inv
    return bool(np.any(ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (s > 1e-6) & (s < s_max)))


def window(t, t0, t1, fade):
    if not (t0 <= t <= t1):
        return 0.0
    return min(1.0, (t - t0) / fade, (t1 - t) / fade)


class Film:
    def __init__(self, meta, verts):
        self.meta, self.verts = meta, verts
        self.offsets = {k: [] for k in meta}          # label -> [[vec, t0, t1, turns, spin centre]]
        self.theme = "light"
        self.t = 0.0
        self.appear = {}                              # label -> time it becomes visible again
        self.base = set()
        self.cam_keys = []                            # (t, lookat, dist, az, el)
        self.captions = []                            # (t0, t1, step, title, detail)
        self.labels = []                              # (t0, t1, [(text | f(t), labels | f(t)->xyz, (dx, dy))])
        self.cards = []                               # (t0, t1, big, small)
        self.readouts = []                            # (t0, t1, [stage])
        self.explode = (0.0, 0.0)
        self.implode = None
        self.hide_at = None                           # all non-base parts vanish here
        self.rope = []                                # (stage, t0, t1) drawing-in
        self.rope_fade = (None, None)
        self.rope_return = None
        self.beads = []                               # (t0, t1)
        self.highlight = []                           # (labels, t0, t1)
        self.ghost = []                               # (labels, t0, t1, alpha)
        self.motion = []                              # (stage, t0, t1, amplitude deg, cycles)
        self.desk = None
        self.desk_ghost = None                        # (t0, t1): desk fades back for close-ups
        self.floor_ghost = []                         # [(t0, t1)]: floor fades for parts fitted from below
        self.duration = 0.0
        self.hidden = {k for k in meta if k.startswith(("Leg", "Stand_", "Rail_Carriage", "REF_MGN12H", "Carriage_HW",
                                                        "Abad_", "Hip_"))}   # leg / 3-DOF films only
        self.abad_fn = None                           # f(t) -> ab/ad angle (rad) of the module about X (3-DOF film)
        self.abad_rope_t = -1.0                       # the ab/ad capstan's rope is drawn from this time
        self.custom = {}                              # label -> f(t) -> (R, p) in model coordinates
        self.angle_fn = None                          # f(t) -> {"A": rad, "B": rad}, overrides motion
        self.up_at = None                             # f(t) -> (R, T): model -> world, overrides UP_R/UP_T
        self.world_cam = False                        # camera keys are already in world coordinates
        self.extras = None                            # f(scn, t): extra scene geometry (e.g. a rail)
        self.reach = {}                               # label -> longer approach than APPROACH
        self.tris = None                              # label -> (N, 3, 3) triangles at rest, model frame

    # --- authoring --------------------------------------------------------------------------
    def offset(self, labels, vec, turns=0.0, center=None):
        """Exploded offset; with turns, the part also spins about the offset axis (threads in), through
        `center` (a point on the screw's axis) or else its bounding-box centre."""
        for k in labels:
            self.offsets[k].append([np.array(vec, float), None, None, turns, center])

    def move(self, labels, t0, t1):
        """Remove the innermost still-pending offset of each label over [t0, t1]."""
        for k in labels:
            o = [o for o in self.offsets[k] if o[1] is None][0]
            o[1], o[2] = t0, t1
            self.appear[k] = min(self.appear.get(k, 1e9), t0 - 0.5)

    def cam(self, t, lookat, dist, az, el):
        self.cam_keys.append((t, np.array(lookat, float), float(dist), float(az), float(el)))

    def frame(self, labels, t, az, el, margin=1.25, extra_t=None):
        pts = []
        for tt in (t,) + ((extra_t,) if extra_t is not None else ()):
            for k in labels:
                pts.append(self.verts[k][[0, 1]] + self.offset_at(k, tt))
        pts = np.vstack(pts)
        lo, hi = pts.min(0), pts.max(0)
        radius = np.linalg.norm(hi - lo) / 2
        dist = 1.15 * margin * radius / math.sin(math.radians(FOVY / 2))
        return (lo + hi) / 2, max(dist, 120.0), az, el

    def clear_view(self, targets, cam, t, t0, t1, skip=(), level=0.15, host_level=0.4):
        """Fade, over [t0, t1], every part seen at time t between the camera (lookat, dist, az, el;
        model frame, world angles) and any of the target points: what is being fitted stays in sight."""
        if not self.tris:
            return []
        lookat, dist, az, el = cam
        a_, e_ = math.radians(az), math.radians(el)
        f = np.array([math.cos(e_) * math.cos(a_), math.cos(e_) * math.sin(a_), math.sin(e_)])
        R = self.up_at(t)[0] if self.up_at is not None else UP_R
        o = np.asarray(lookat, float) - dist * (R.T @ f)
        skip, hits = set(skip), set()
        for k, tri in self.tris.items():
            if k in skip or self.alpha(k, t) < 0.5:
                continue
            off = self.offset_at(k, t)
            lo, hi = self.verts[k][0] + off - 0.5, self.verts[k][1] + off + 0.5
            for p in targets:
                d = np.asarray(p, float) - o
                with np.errstate(divide="ignore", invalid="ignore"):
                    ta, tb = (lo - o) / d, (hi - o) / d
                tmin = np.nanmax(np.minimum(ta, tb))
                tmax = np.nanmin(np.maximum(ta, tb))
                if tmax < max(tmin, 0.0) or tmin > 1.0:
                    continue
                if ray_hits(tri + off, o, d, 1.0 - 6.0 / np.linalg.norm(d)):
                    hits.add(k)
                    break
        # a part the hardware ends up inside (its host) only half-fades: it stays readable
        inside = lambda k: any(np.all(self.verts[k][0] + self.offset_at(k, t) - 1.0 <= p) and
                               np.all(p <= self.verts[k][1] + self.offset_at(k, t) + 1.0) for p in targets)
        hosts = sorted(k for k in hits if inside(k))
        rest = sorted(hits - set(hosts))
        if rest:
            self.ghost.append((rest, t0, t1, level))
        if hosts:
            self.ghost.append((hosts, t0, t1, max(level, host_level)))
        return sorted(hits)

    # --- evaluation -------------------------------------------------------------------------
    def explode_factor(self, t):
        if self.hide_at is not None and t >= self.hide_at:
            return 1.0
        if self.implode and t >= self.implode[0]:
            return 1.0 - ease((t - self.implode[0]) / (self.implode[1] - self.implode[0]))
        t0, t1 = self.explode
        return ease((t - t0) / (t1 - t0)) if t1 > t0 else 1.0

    def offset_at(self, k, t):
        e = self.explode_factor(t)
        total = np.zeros(3)
        assembling = self.hide_at is not None and t >= self.hide_at
        for vec, t0, t1, _, _ in self.offsets[k]:
            f = ease((t - t0) / (t1 - t0)) if (t0 is not None and assembling) else 0.0
            total += vec * e * (1 - f)
        if assembling:
            n = np.linalg.norm(total)
            a = self.reach.get(k, APPROACH)
            if n > a:                                 # parts approach from close by, not from the exploded view
                total *= a / n
        return total

    def pose_at(self, k, t):
        """(R, p): world point x of part k maps to R x + p at time t."""
        R = np.eye(3)
        c = self.verts[k][2]
        if self.hide_at is not None and t >= self.hide_at:
            for vec, t0, t1, turns, center in self.offsets[k]:
                if center is not None:
                    c = np.asarray(center, float)
                if turns and t0 is not None and t < t1:
                    R = axis_rot(vec, -2 * math.pi * turns * (1 - ease((t - t0) / (t1 - t0)))) @ R
        return R, c - R @ c + self.offset_at(k, t)

    def step(self, num, head, detail, groups, az, el, margin=1.25, context=(), hold=0.8, stagger=0.45,
             move_time=1.3, ghost=(), lead=0.8, frame_context=False):
        """One captioned step: each group moves in turn while the camera frames them."""
        start = self.t
        t_move = start + lead
        ends = []
        for i, g in enumerate(groups):
            a = t_move + i * stagger
            self.move(g, a, a + move_time)
            ends.append(a + move_time)
        end = max(ends)
        moving = [k for g in groups for k in g]
        for k in list(context) + list(ghost):
            if k not in self.base:
                self.appear[k] = min(self.appear.get(k, 1e9), start)
        framed = moving + list(ghost) + (list(context) if frame_context else [])
        c0 = self.frame(framed, t_move + 0.5 * move_time, az, el, margin, extra_t=end + 0.01)
        self.cam(start + lead - 0.1, *c0)
        self.cam(end + hold, c0[0], c0[1], c0[2] + 6, c0[3])
        self.highlight.append((moving, t_move - 0.5, end + 0.3))
        if ghost:
            self.ghost.append((list(ghost), start, end + 0.4, 0.4))
        self.captions.append((start + 0.15, end + hold - 0.1, str(num), head, detail))
        self.t = end + hold
        return start, end

    def alpha(self, k, t, ghosts=True):
        """Opacity at t; ghosts=False: whether it is physically there (ignoring the camera's see-through fades)."""
        if k in self.hidden:
            return 0.0
        a = 1.0
        if k not in self.base and self.hide_at is not None:
            if t < self.hide_at:
                a = min(1.0, (self.hide_at - t) / 0.5)
            else:
                ap = self.appear.get(k)
                a = 0.0 if ap is None or t < ap else min(1.0, (t - ap) / 0.4)
        for labs, g0, g1, level in (self.ghost if ghosts else ()):
            if k in labs:
                a = min(a, 1.0 - (1.0 - level) * window(t, g0, g1, 0.5))
        return a

    def camera(self, t):
        keys = self.cam_keys
        if t <= keys[0][0]:
            return keys[0][1:]
        for (ta, la, da, aa, ea), (tb, lb, db, ab, eb) in zip(keys, keys[1:]):
            if ta <= t <= tb:
                u = ease((t - ta) / (tb - ta)) if tb > ta else 1.0
                return (la + (lb - la) * u, math.exp(math.log(da) + (math.log(db) - math.log(da)) * u),
                        aa + (ab - aa) * u, ea + (eb - ea) * u)
        return keys[-1][1:]

    def angles(self, t):
        """Output angle (rad) of each big drum; the motors turn -RATIO times as far."""
        if self.angle_fn is not None:
            return self.angle_fn(t)
        th = {"A": 0.0, "B": 0.0}
        for stage, t0, t1, amp, cycles in self.motion:
            if t0 <= t <= t1:
                u = (t - t0) / (t1 - t0)
                env = math.sin(math.pi * u) ** 0.5
                th[stage] += math.radians(amp) * env * math.sin(2 * math.pi * cycles * u)
        return th


# ---------------------------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------------------------
def load_font(size, weight="Regular"):
    path = "/System/Library/Fonts/Avenir Next.ttc"
    if os.path.exists(path):
        for i in range(12):
            try:
                f = ImageFont.truetype(path, size, index=i)
            except OSError:
                break
            if f.getname()[1] == weight:
                return f
    for p in ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F_TITLE = load_font(38, "Demi Bold")
F_BODY = load_font(25, "Regular")
F_SMALL = load_font(22, "Medium")
F_STEP = load_font(26, "Bold")
F_LABEL = load_font(23, "Demi Bold")
F_HERO = load_font(84, "Bold")
F_NUM = load_font(44, "Demi Bold")


def project(points, lookat, dist, az, el, w=None, h=None):
    az, el = math.radians(az), math.radians(el)
    f = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    pos = lookat - dist * f
    right = np.cross(f, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    rel = np.atleast_2d(points) - pos
    depth = rel @ f
    w, h = w or W, h or H
    s = (h / 2) / math.tan(math.radians(FOVY / 2))
    return np.stack([w / 2 + s * (rel @ right) / depth, h / 2 - s * (rel @ up) / depth], axis=1)


def overlay(img, film, t, cam, title, th):
    base = img.convert("RGBA")
    layer = Image.new("RGBA", (OW, OH), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    ink = THEMES[film.theme]["ink"]
    d.text((48, 36), title, font=F_SMALL, fill=ink + (200,))

    for t0, t1, step, head, detail in film.captions:
        a = window(t, t0, t1, 0.35)
        if a <= 0:
            continue
        x0, y1 = 48, OH - 48
        tx = x0 + (96 if step else 26)
        w = max(d.textlength(head, font=F_TITLE), d.textlength(detail, font=F_BODY) if detail else 0) + tx - x0 + 30
        h = 112 if detail else 76
        d.rounded_rectangle((x0, y1 - h, x0 + w, y1), 16, fill=(24, 30, 40, int(214 * a)))
        if step:
            d.rounded_rectangle((x0 + 20, y1 - h + 18, x0 + 76, y1 - h + 58), 10, fill=(214, 72, 52, int(255 * a)))
            d.text((x0 + 48, y1 - h + 38), step, font=F_STEP, fill=(255, 255, 255, int(255 * a)), anchor="mm")
        d.text((tx, y1 - h + 12), head, font=F_TITLE, fill=(255, 255, 255, int(255 * a)))
        if detail:
            d.text((tx, y1 - h + 66), detail, font=F_BODY, fill=(206, 213, 222, int(255 * a)))

    for t0, t1, items in film.labels:
        a = window(t, t0, t1, 0.45)
        if a <= 0:
            continue
        placed = []
        for text, where, (dx, dy) in items:
            text = text(t) if callable(text) else text
            if callable(where):
                p3 = where(t)
            else:
                p3 = np.vstack([film.verts[k][2] + film.offset_at(k, t) for k in where]).mean(0)
            px = project(to_world(p3), *cam, w=OW, h=OH)[0]
            lx, ly = px[0] + dx, px[1] + dy
            tw = d.textlength(text, font=F_LABEL)
            box = [lx - tw / 2 - 14, ly - 20, lx + tw / 2 + 14, ly + 20]
            for other in placed:                      # nudge away from earlier labels
                if box[0] < other[2] and box[2] > other[0] and box[1] < other[3] and box[3] > other[1]:
                    shift = other[3] - box[1] + 6 if dy >= 0 else other[1] - box[3] - 6
                    box[1] += shift; box[3] += shift; ly += shift
            placed.append(box)
            d.line((px[0], px[1], lx, ly), fill=(30, 36, 46, int(170 * a)), width=2)
            d.ellipse((px[0] - 5, px[1] - 5, px[0] + 5, px[1] + 5), fill=(214, 72, 52, int(255 * a)))
            d.rounded_rectangle(box, 10, fill=(255, 255, 255, int(232 * a)), outline=(30, 36, 46, int(90 * a)))
            d.text((lx, ly), text, font=F_LABEL, fill=(30, 36, 46, int(255 * a)), anchor="mm")

    for t0, t1, stages in film.readouts:          # live motor vs output angle
        a = window(t, t0, t1, 0.4)
        if a <= 0:
            continue
        bw, row = 470, 66
        x1, y1 = OW - 48, OH - 48
        y0 = y1 - 54 - row * len(stages)
        d.rounded_rectangle((x1 - bw, y0, x1, y1), 16, fill=(24, 30, 40, int(214 * a)))
        d.text((x1 - bw + 26, y0 + 14), "MOTOR", font=F_SMALL, fill=(170, 180, 192, int(255 * a)))
        d.text((x1 - bw + 250, y0 + 14), "OUTPUT", font=F_SMALL, fill=(170, 180, 192, int(255 * a)))
        for i, s in enumerate(stages):
            y = y0 + 48 + row * i
            out = math.degrees(th[s])
            col = (232, 96, 76) if s == "A" else (238, 190, 60) if s == "C" else (150, 160, 175)
            d.text((x1 - bw + 26, y), f"{s}", font=F_NUM, fill=col + (int(255 * a),))
            d.text((x1 - bw + 64, y), f"{-RATIO * out:+6.0f}°", font=F_NUM, fill=(255, 255, 255, int(255 * a)))
            d.text((x1 - bw + 250, y), f"{out:+5.1f}°", font=F_NUM, fill=(255, 255, 255, int(255 * a)))

    for t0, t1, big, small in film.cards:
        a = window(t, t0, t1, 0.6)
        if a <= 0:
            continue
        d.text((OW / 2, OH - 190), big, font=F_HERO, fill=ink + (int(255 * a),), anchor="mm")
        d.text((OW / 2, OH - 115), small, font=F_BODY, fill=ink + (int(200 * a),), anchor="mm")
    if layer.size != base.size:
        layer = layer.resize(base.size, Image.LANCZOS)
    return Image.alpha_composite(base, layer).convert("RGB")


# ---------------------------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------------------------
def explode_labels(L):
    names = [("Motor plate + 8308 motors", L("Motor_Plate")), ("moteus r4.11", L("REF_moteus")),
             ("Small drums", L("Small_Drum")), ("Posts", L("Post_1", "Post_2")),
             ("Big drum A", L("Big_Drum_A")), ("Base mount", L("Base_Mount")), ("Big drum B", L("Big_Drum_B")),
             ("Front wall + 6808", L("Front_Wall")), ("Output flange B", L("Output_Flange_B")),
             ("Test arm B", L("Test_Arm_B")), ("Output flange A", L("Output_Flange_A")),
             ("Test arm A", L("Test_Arm_A"))]
    return [(text, labs, (0, -170 if i % 2 else 190)) for i, (text, labs) in enumerate(names)]


def set_offsets(F, L):
    """Exploded offsets. Parts fitted into a sub-assembly first carry their own offset, then the host's."""
    y = lambda dy: (0, dy, 0)
    G = {"small": 110, "drumA": 190, "axle": 270, "drumB": 340, "wall": 450, "flB": 560, "armB": 660,
         "flA": 750, "armA": 840}
    F.offset(L("Small_Drum"), y(G["small"]))
    F.offset(L("FST_Rotor"), y(G["small"] + 45))
    F.offset(L("Post"), y(150))
    F.offset(L("REF_Bearing_A_30x42_rear"), y(-60))
    F.offset(L("REF_Bearing_A_30x42_front"), y(70))
    F.offset(L("FST_Insert_A_shaft"), y(-25))
    for side, tag, sgn in (("1", "minus", -1), ("2", "plus", 1)):
        F.offset(L(f"Rope_Tensioner_Slider_{side}", f"REF_Tensioner_A_{tag}_nut"), y(34 * sgn))
    drum_a = L("Big_Drum_A", "Rope_Tensioner_Slider_1", "Rope_Tensioner_Slider_2", "REF_Tensioner_A",
               "FST_Insert_A_shaft")
    F.offset(drum_a, y(G["drumA"]))
    F.offset(L("A_Shaft_Cap"), y(-60))
    F.offset(L("FST_Screw_A_cap"), y(-95))
    F.offset(L("REF_Bearing_30x42_B1"), y(-55))
    F.offset(L("REF_Bearing_30x42_B2"), y(-90))                 # both from the rear, B2 first
    F.offset(L("FST_Insert_B_hub"), y(30))
    for side, tag, sgn in (("3", "minus", -1), ("4", "plus", 1)):
        F.offset(L(f"Rope_Tensioner_Slider_{side}", f"REF_Tensioner_B_{tag}_nut"), y(34 * sgn))
    drum_b = L("Big_Drum_B", "REF_Bearing_30x42", "FST_Insert_B_hub", "Rope_Tensioner_Slider_3",
               "Rope_Tensioner_Slider_4", "REF_Tensioner_B")
    F.offset(drum_b, y(G["drumB"]))
    for k in L("REF_Tensioner_A", "REF_Tensioner_B"):         # tension screws go in from the drum ends
        if k.endswith("_screw"):
            st_, s_ = ts_of(k)
            o, xd, zd = ts_frame(st_, s_)
            F.offset([k], -30.0 * xd, turns=5, center=ts_point(st_, s_, TS_SPOT, TS_ZS))
    F.offset(L("REF_Bearing_15x28_tip"), y(40))
    F.offset(L("REF_Bearing_40x52_wall"), y(-40))              # into the wall from the rear
    F.offset(L("Front_Wall", "REF_Bearing_15x28_tip", "REF_Bearing_40x52_wall"), y(G["wall"]))
    F.offset(L("FST_Bolt_posts"), y(G["wall"] + 50))
    F.offset(L("FST_Nut_posts"), y(-60))
    F.offset(L("FST_Nut_B_flange"), y(-22))
    F.offset(L("Output_Flange_B", "FST_Nut_B_flange"), y(G["flB"]))
    F.offset(L("FST_Screw_B_flange"), y(G["flB"] + 40))
    F.offset(L("Test_Arm_B"), y(G["armB"]))
    F.offset(L("FST_Screw_B_arm"), y(G["armB"] + 40))
    F.offset(L("FST_Insert_A_nose"), y(25))
    F.offset(L("FST_Insert_A_flange"), y(25))
    F.offset(L("Output_Flange_A", "FST_Insert_A_flange"), y(G["flA"]))
    F.offset(L("FST_Screw_A_flange"), y(G["flA"] + 40))
    F.offset(L("Test_Arm_A"), y(G["armA"]))
    F.offset(L("FST_Screw_A_arm"), y(G["armA"] + 40))
    F.offset(L("FST_Insert_ribs"), (0, 0, 40))                  # model +Z is down on the desk
    F.offset(L("Base_Mount"), (0, -220, 0))                      # slides in from behind
    F.offset(L("FST_Screw_base"), (0, 0, 40))
    return drum_a, drum_b


EXPLODED_CAM = (np.array([0, 390, -50]), 1600, -28, -12)
HERO = (np.array([0, 50, -95]), 900, -128, -14)


def author_assembly(meta, verts):
    F = Film(meta, verts)
    L = lambda *n: labels_of(meta, *n)
    base = L("Motor_Plate", "REF_Motor", "REF_moteus", "FST_Stator", "FST_Insert_moteus", "FST_Screw_moteus", "FST_Standoff_moteus")
    F.base = set(base)
    plate = L("Motor_Plate")
    drum_a, drum_b = set_offsets(F, L)

    # --- intro: assembled orbit while both outputs swing -----------------------------------
    F.cards.append((0.3, 4.6, "Coaxial Capstan v3", "Two 8308 motors  ·  8:1 rope capstans  ·  coaxial outputs"))
    F.cam(0.0, *HERO)
    F.cam(5.0, HERO[0], 860, -100, -16)
    F.motion += [("A", 0.4, 5.0, 14, 1.0), ("B", 0.4, 5.0, -10, 1.0)]
    F.rope_fade = (5.0, 5.5)
    # --- explode ----------------------------------------------------------------------------
    F.explode = (5.5, 8.3)
    F.cam(8.3, EXPLODED_CAM[0], EXPLODED_CAM[1], -30, -12)
    F.cam(11.8, EXPLODED_CAM[0], EXPLODED_CAM[1], -16, -10)
    F.captions.append((5.7, 11.6, "", "Exploded view", "Front to back in insertion order"))
    F.labels.append((8.4, 11.6, explode_labels(L)))
    F.hide_at = 12.4
    t = 12.4

    # --- step-by-step assembly --------------------------------------------------------------
    def step(*args, **kw):
        nonlocal t
        F.t = t
        out = F.step(*args, **kw)
        t = F.t
        return out

    F.cam(t, np.array([0, 20, -30]), 560, -125, -20)
    F.captions.append((t + 0.15, t + 2.0, "", "Motor plate", "8308 motors on 4× M4 countersunk  ·  moteus r4.11 behind"))
    F.cam(t + 2.1, np.array([0, 20, -30]), 540, -108, -18)
    t += 2.1

    step(1, "Small drums", "4× M3×10 each into the rotor",
         [L("Small_Drum"), L("FST_Rotor")], -120, -18, margin=1.0, stagger=1.0)
    step(2, "Frame posts", "Five posts, clear of every swept path",
         [L("Post")], -125, -14, context=plate, stagger=0.1)
    step(3, "Drum A bearings", "2× 6806 (30×42×7) into the plate housing, one from each side, 21.5 mm apart",
         [L("REF_Bearing_A_30x42_rear"), L("REF_Bearing_A_30x42_front")], -150, -14, margin=1.6,
         context=plate, stagger=0.8)
    step(3, "Drum A tensioners", "Slider with its M3 nut drops into each end pocket",
         [L("Rope_Tensioner_Slider_2", "REF_Tensioner_A_plus_nut"), L("Rope_Tensioner_Slider_1", "REF_Tensioner_A_minus_nut")],
         -150, -22, margin=1.15, context=L("Big_Drum_A"), stagger=0.7)
    step(3, "Drum A into the plate", "Its Ø30 shaft slides through both bearings", [drum_a], -125, -14,
         margin=1.05, context=plate)
    step(3, "Rear cap", "Printed cap + M5×12 into the shaft's insert, from the back",
         [L("A_Shaft_Cap"), L("FST_Screw_A_cap")], 150, -12, margin=1.6, context=L("A_Shaft_Cap"), stagger=0.9)
    step(4, "Drum B bearings + inserts", "2× 6806 from the rear  ·  6× M3 inserts in the journal end  ·  two tensioners",
         [L("REF_Bearing_30x42_B2"), L("REF_Bearing_30x42_B1"), L("FST_Insert_B_hub"),
          L("Rope_Tensioner_Slider_4", "REF_Tensioner_B_plus_nut"), L("Rope_Tensioner_Slider_3", "REF_Tensioner_B_minus_nut")],
         -140, -15, margin=1.1, ghost=L("Big_Drum_B"), stagger=0.6, move_time=1.1)
    step(4, "Drum B onto A's tube", "", [drum_b], -125, -14, margin=1.0, context=L("Big_Drum_A"))

    start = t                                                   # ropes, then tension from the drum ends
    set_a = [k for k in L("REF_Tensioner_A") if k.endswith("_screw")]
    set_b = [k for k in L("REF_Tensioner_B") if k.endswith("_screw")]
    F.cam(start + 0.8, np.array([-30, 50, -40]), 400, -40, -45)
    F.rope.append(("A", start + 1.0, start + 4.2))
    F.move(set_a, start + 4.2, start + 5.6)
    F.highlight.append((set_a, start + 3.8, start + 6.0))
    F.cam(start + 5.9, np.array([-30, 50, -40]), 390, -55, -48)
    F.cam(start + 6.5, np.array([30, 72, -40]), 400, -140, -45)
    F.rope.append(("B", start + 6.6, start + 9.8))
    F.move(set_b, start + 9.8, start + 11.2)
    F.highlight.append((set_b, start + 9.4, start + 11.6))
    F.cam(start + 11.6, np.array([20, 65, -40]), 440, -120, -42)
    F.captions.append((start + 0.15, start + 4.1, "5", "Ropes", "Crossed loop, ~5 turns on each small drum"))
    F.captions.append((start + 4.2, start + 11.5, "5", "Tension from the drum ends", "M3×12 in from each end face pulls its slider and rope  ·  2.5 mm hex key"))
    for k in set_a + set_b:                                     # hidden until they are driven in
        F.appear[k] = start + 3.7 if k in set_a else start + 9.3
    t = start + 11.6

    step(6, "Front wall bearings", "2× 6902 for the small-drum tips  ·  6808 for drum B's journal",
         [L("REF_Bearing_15x28_tip"), L("REF_Bearing_40x52_wall")], -140, -14, margin=1.1,
         ghost=L("Front_Wall"), stagger=0.6)
    step(6, "Front wall", "The 6808 slides onto drum B's journal  ·  5× M4 through the posts",
         [L("Front_Wall", "REF_Bearing_15x28_tip", "REF_Bearing_40x52_wall"), L("FST_Bolt_posts"), L("FST_Nut_posts")],
         -128, -12, margin=0.95, context=L("Big_Drum_B"), stagger=1.0)
    step(7, "Output flange B", "6× M3 nuts captured, 6× M3×16 into the journal; clamps the 6808",
         [L("FST_Nut_B_flange"), L("Output_Flange_B", "FST_Nut_B_flange"), L("FST_Screw_B_flange")],
         -140, -10, margin=1.15, context=L("Big_Drum_B"), ghost=L("Output_Flange_B"), stagger=0.9)
    step(7, "Test arm B", "6× M3×12 into the flange nuts",
         [L("Test_Arm_B"), L("FST_Screw_B_arm")], -130, -10, margin=1.0, context=L("Output_Flange_B"), stagger=1.0)
    step(8, "Output flange A", "4× M3 inserts in A's nose, 4× M3×10  ·  6× M3 inserts for the arm",
         [L("FST_Insert_A_nose"), L("FST_Insert_A_flange"), L("Output_Flange_A", "FST_Insert_A_flange"), L("FST_Screw_A_flange")],
         -150, -6, margin=1.3, context=L("Output_Flange_B"), ghost=L("Output_Flange_A"), stagger=0.9)
    step(8, "Test arm A", "6× M3×10  ·  M5 load holes at 50–150 mm",
         [L("Test_Arm_A"), L("FST_Screw_A_arm")], -130, -10, margin=0.95, context=L("Output_Flange_B"), stagger=1.0)
    step(9, "Base mount", "4× M4 rib inserts, 4× M4×12 from below, clamp to the desk",
         [L("FST_Insert_ribs"), L("Base_Mount"), L("FST_Screw_base")], -118, -14, margin=0.9,
         context=plate + L("Front_Wall"), stagger=0.9)
    F.desk = (t - 0.4, t + 0.6)

    drivetrain(F, L, t)
    return F


def drivetrain(F, L, t):
    """Frame turns see-through; follow each stage from motor rotor to output arm; both together."""
    frame = L("Front_Wall", "Post", "FST_Bolt_posts", "FST_Nut_posts", "Base_Mount", "FST_Screw_base",
              "FST_Insert_ribs", "REF_Bearing_15x28_tip", "REF_Bearing_40x52_wall")
    s0 = t
    T = 16.0
    F.ghost.append((frame, s0 + 0.3, s0 + 2 * T + 7.0, 0.12))
    F.desk_ghost = (s0 + 0.3, s0 + 2 * T + 6.0)
    F.motion.append(("A", s0 + 1.0, s0 + T, 20, 2.0))
    F.beads.append((s0 + 0.8, s0 + 2 * T + 7.0))
    F.readouts.append((s0 + 1.0, s0 + T, ["A"]))
    rotor_a = L("REF_Motor_A_rotor", "Small_Drum_A", "FST_Rotor_A")
    big_a = L("Big_Drum_A", "Rope_Tensioner_Slider_1", "Rope_Tensioner_Slider_2", "REF_Tensioner_A")
    out_a = L("Output_Flange_A", "Test_Arm_A", "FST_Screw_A", "FST_Insert_A_flange", "FST_Insert_A_nose")
    look_m = np.array([MA[0], 45, MA[1] - 5])
    F.cam(s0 + 1.0, look_m, 320, -160, -18)    # upright: side views past the raised arms
    F.highlight += [(rotor_a, s0 + 1.0, s0 + 5.5), (big_a, s0 + 5.0, s0 + 10.0), (out_a, s0 + 9.5, s0 + 14.5)]
    F.labels.append((s0 + 1.2, s0 + 5.4, [("Motor A rotor + small drum", lambda t: np.array([MA[0], 40, MA[1] + 32]), (-60, -150))]))
    F.cam(s0 + 5.0, look_m, 320, -170, -20)
    F.cam(s0 + 6.3, np.array([-25, 52, -45]), 380, -165, -25)
    F.labels.append((s0 + 6.0, s0 + 10.0, [
        ("Rope: 8:1 onto big drum A", lambda t: rope_point("A", F.angles(t)["A"], 0.6), (-170, -120)),
        ("Big drum A", lambda t: rim_point("A", F.angles(t)["A"]), (180, -40))]))
    F.cam(s0 + 9.6, np.array([-25, 52, -45]), 370, -155, -25)
    F.cam(s0 + 10.9, np.array([0, 110, -120]), 600, -135, -10)
    F.labels.append((s0 + 10.6, s0 + T, [("Output A: flange + arm", lambda t: arm_point("A", F.angles(t)["A"], 100), (190, 40))]))
    F.cam(s0 + T, np.array([0, 110, -120]), 600, -120, -10)
    F.captions.append((s0 + 0.3, s0 + T, "", "Drive A: motor to drum to arm", "Motor turns 8× the output, through one rope loop"))

    s1 = s0 + T
    T2 = 11.0
    F.motion.append(("B", s1 + 0.5, s1 + T2, 20, 1.5))
    F.readouts.append((s1 + 0.5, s1 + T2, ["B"]))
    rotor_b = L("REF_Motor_B_rotor", "Small_Drum_B", "FST_Rotor_B")
    big_b = L("Big_Drum_B", "Rope_Tensioner_Slider_3", "Rope_Tensioner_Slider_4", "REF_Tensioner_B")
    out_b = L("Output_Flange_B", "Test_Arm_B", "FST_Screw_B", "FST_Nut_B", "FST_Insert_B")
    F.cam(s1 + 1.0, np.array([25, 65, -30]), 380, -20, -25)
    F.highlight += [(rotor_b, s1 + 0.8, s1 + 4.0), (big_b, s1 + 3.5, s1 + 7.0), (out_b, s1 + 6.5, s1 + 10.5)]
    F.labels.append((s1 + 1.2, s1 + 6.5, [
        ("Motor B rotor + small drum", lambda t: np.array([MB[0], 40, MB[1] + 32]), (60, -150)),
        ("Big drum B", lambda t: rim_point("B", F.angles(t)["B"]), (-180, -40))]))
    F.cam(s1 + 6.0, np.array([25, 65, -30]), 370, -30, -25)
    F.cam(s1 + 7.4, np.array([0, 110, -120]), 600, -55, -10)
    F.labels.append((s1 + 7.2, s1 + T2, [("Output B: outer flange + arm", lambda t: arm_point("B", F.angles(t)["B"], 100), (-200, 40))]))
    F.cam(s1 + T2, np.array([0, 110, -120]), 600, -70, -10)
    F.captions.append((s1 + 0.2, s1 + T2, "", "Drive B", "Same 8:1, on the outer flange, independent of A"))

    s2 = s1 + T2
    F.motion += [("A", s2 + 0.3, s2 + 6.0, 18, 1.5), ("B", s2 + 0.3, s2 + 6.0, -15, 1.0)]
    F.readouts.append((s2 + 0.3, s2 + 6.0, ["A", "B"]))
    F.cam(s2 + 1.0, np.array([0, 80, -100]), 720, -130, -15)
    F.cam(s2 + 6.5, HERO[0], 860, -95, -16)
    F.captions.append((s2 + 0.2, s2 + 6.0, "", "Both outputs, one axis", ""))
    F.cards.append((s2 + 6.4, s2 + 9.0, "Coaxial Capstan v3", "CAD animation  ·  threads omitted"))
    F.duration = s2 + 9.0


def show_part(F, count, labs, head, detail, az):
    """Short orbit around one printed part on its own before its sub-assembly."""
    st = F.t
    for k in labs:
        F.appear[k] = min(F.appear.get(k, 1e9), st)
    look, dist, _, _ = F.frame(labs, st, az, -15, 1.2)
    F.cam(st + 0.6, look, dist, az, -15)
    F.cam(st + 3.2, look, dist, az + 70, -22)
    F.highlight.append((labs, st + 0.2, st + 3.0))
    F.captions.append((st + 0.15, st + 3.1, str(next(count)), head, detail))
    F.t = st + 3.3


def author_detailed(meta, verts):
    """Kit-style build: every part and fastener, one action per step, screws threading in."""
    F = Film(meta, verts)
    F.theme = "dark"
    L = lambda *n: labels_of(meta, *n)
    y = lambda dy: (0, dy, 0)
    plate = L("Motor_Plate")
    F.base = set(plate)
    motor = lambda w: L(f"REF_Motor_{w}_stator", f"REF_Motor_{w}_rotor")
    holder = L("REF_Motor_A_magnet_holder", "REF_Motor_B_magnet_holder")
    magnet = L("REF_Motor_A_magnet_disc", "REF_Motor_B_magnet_disc")
    motors = motor("A") + motor("B") + holder + magnet

    # offsets, innermost (first fitted) first; screws carry turns so they thread in
    F.offset(magnet, y(-22))
    F.offset(holder + magnet, y(-45))
    F.offset(motors, y(160))
    F.offset(L("FST_Stator"), y(-28), turns=6)
    F.offset(L("FST_Insert_moteus"), y(-22))
    F.offset(L("REF_moteus"), y(-60))
    F.offset(L("FST_Standoff_moteus"), y(-26), turns=6)
    F.offset(L("FST_Screw_moteus"), y(-24), turns=5)
    F.offset(L("Small_Drum"), y(240))
    F.offset(L("FST_Rotor"), y(30), turns=5)
    F.offset(L("FST_Rotor"), y(240))
    F.offset(L("Post"), y(330))
    F.offset(L("REF_Bearing_A_30x42_rear"), y(-60))
    F.offset(L("REF_Bearing_A_30x42_front"), y(70))
    F.offset(L("FST_Insert_A_shaft"), y(-25))
    for side, tag, sgn in (("1", "minus", -1), ("2", "plus", 1)):
        F.offset(L(f"Rope_Tensioner_Slider_{side}"), y(34 * sgn))
        F.offset(L(f"REF_Tensioner_A_{tag}_nut"), y(30 * sgn))
    drum_a = L("Big_Drum_A", "Rope_Tensioner_Slider_1", "Rope_Tensioner_Slider_2", "REF_Tensioner_A",
               "FST_Insert_A_shaft")
    F.offset(drum_a, y(420))
    F.offset(L("A_Shaft_Cap"), y(-45))
    F.offset(L("FST_Screw_A_cap"), y(-30), turns=6)
    F.offset(L("REF_Bearing_30x42_B1"), y(-55))
    F.offset(L("REF_Bearing_30x42_B2"), y(-90))                 # both from the rear, B2 first
    F.offset(L("FST_Insert_B_hub"), y(25))
    for side, tag, sgn in (("3", "minus", -1), ("4", "plus", 1)):
        F.offset(L(f"Rope_Tensioner_Slider_{side}"), y(34 * sgn))
        F.offset(L(f"REF_Tensioner_B_{tag}_nut"), y(30 * sgn))
    drum_b = L("Big_Drum_B", "REF_Bearing_30x42", "FST_Insert_B_hub", "Rope_Tensioner_Slider_3",
               "Rope_Tensioner_Slider_4", "REF_Tensioner_B")
    F.offset(drum_b, y(560))
    for k in L("REF_Tensioner_A", "REF_Tensioner_B"):          # tension screws, along the rope line
        if k.endswith("_screw"):
            st_, s_ = ts_of(k)
            o, xd, zd = ts_frame(st_, s_)
            c = ts_point(st_, s_, TS_SPOT, TS_ZS)
            F.offset([k], -24.0 * xd, turns=5, center=c)             # in from the end face
            F.offset([k], -0.01 * xd, turns=3, center=c)             # tightening: turns in place
    for (st_, s_), sl in SLIDER.items():                          # sliders sit slack until tightened
        o, xd, zd = ts_frame(st_, s_)
        F.offset([sl] + L(f"REF_Tensioner_{st_}_{'plus' if s_ > 0 else 'minus'}_nut"), TS_TRAVEL * xd)
    F.offset(L("REF_Bearing_15x28_tip"), y(40))
    F.offset(L("REF_Bearing_40x52_wall"), y(-40))              # into the wall from the rear
    F.offset(L("Front_Wall", "REF_Bearing_15x28_tip", "REF_Bearing_40x52_wall"), y(680))
    F.offset(L("FST_Bolt_posts"), y(60), turns=8)
    F.offset(L("FST_Nut_posts"), y(-30), turns=4)
    F.offset(L("FST_Nut_B_flange"), y(-22))
    F.offset(L("Output_Flange_B", "FST_Nut_B_flange"), y(790))
    F.offset(L("FST_Screw_B_flange"), y(24), turns=5)
    F.offset(L("Test_Arm_B"), y(880))
    F.offset(L("FST_Screw_B_arm"), y(24), turns=5)
    F.offset(L("FST_Insert_A_nose"), y(25))
    F.offset(L("FST_Insert_A_flange"), y(25))
    F.offset(L("Output_Flange_A", "FST_Insert_A_flange"), y(970))
    F.offset(L("FST_Screw_A_flange"), y(26), turns=5)
    F.offset(L("Test_Arm_A"), y(1060))
    F.offset(L("FST_Screw_A_arm"), y(24), turns=5)
    F.offset(L("FST_Insert_ribs"), (0, 0, 40))                  # model +Z is down on the desk
    F.offset(L("Base_Mount"), (0, -220, 0))                      # slides in from behind
    F.offset(L("FST_Screw_base"), (0, 0, 30), turns=5)

    # parts overview, then build up from the bare plate
    F.explode = (0.0, 0.0)
    F.cam(0.0, np.array([0, 480, -60]), 2050, -36, -14)
    F.cam(3.6, np.array([0, 480, -60]), 2000, -24, -12)
    F.captions.append((0.2, 3.4, "", "Coaxial Capstan v3: parts", "Printed parts, motors, moteus boards, bearings and every fastener"))
    F.hide_at = 4.0
    F.t = 4.0
    rear = 50                                                    # azimuth looking at the plate's back
    count = iter(range(1, 100))
    last = [0]

    def S(num, *a, **k):
        """Steps number themselves; num == -1 continues the previous step's number."""
        if num != -1:
            last[0] = next(count)
        return F.step(last[0], *a, **k)
    F.cam(4.0, np.array([0, 0, -30]), 520, -125, -20)

    S(1, "M2.5 heat-set inserts", "Per board: 2× into the printed bosses, 2× flush in the plate back",
      [L(f"FST_Insert_moteus_{w}_{i}") for w in "AB" for i in (1, 2, 3, 4)], rear, -12, margin=1.2,
      stagger=0.1, move_time=0.9)
    S(2, "Encoder magnets into the holders", "Ø8 × 2 mm diametric magnet, bonded into each 8308 magnet holder",
      [magnet], -20, -12, margin=1.3, ghost=holder, move_time=1.1)
    S(3, "Magnet holders onto the rotors", "Holder fits over the rotor's rear hex",
      [holder + magnet], -20, -12, margin=1.2, context=motor("A") + motor("B"))
    S(4, "EaglePower 8308 motors onto the plate", "Phase wires already soldered to the moteus slide into the open side slot",
      [motors], -125, -16, margin=1.0, context=plate, frame_context=True)
    S(5, "Stator screws", "4× M4 × 12 countersunk (DIN 7991) per motor from the back, PCD 40",
      [L(f"FST_Stator_{w}_{i}") for i in (1, 2, 3, 4) for w in "AB"], rear, -14, margin=1.2,
      stagger=0.12, move_time=1.6)
    S(6, "Brass standoffs", "2× M2.5 M-F × 8 mm per board into the flush inserts, after the stator screws",
      [L(f"FST_Standoff_moteus_{w}_{i}") for w in "AB" for i in (1, 2)], rear, -12, margin=1.4,
      stagger=0.2, move_time=1.5)
    S(7, "moteus r4.11 controllers", "Connectors point outboard, the same way as the motor's 3 wires; encoder over the magnet",
      [L("REF_moteus_A"), L("REF_moteus_B")], rear, -10, margin=1.0, stagger=0.4)
    S(8, "Controller screws", "8× M2.5 × 6 socket head into the bosses and standoffs",
      [L(f"FST_Screw_moteus_{w}_{i}") for w in "AB" for i in (1, 2, 3, 4)], rear, -10, margin=1.1,
      context=L("REF_moteus"), stagger=0.12, move_time=1.4)
    S(8, "Small drums onto the rotors", "Helical 2.5 mm pitch groove for Ø2 mm rope",
      [L("Small_Drum_A", "Small_Drum_B", "FST_Rotor")], -125, -18, margin=1.0, context=plate)
    S(9, "Small drum screws", "4× M3 × 10 socket head per drum into the rotor, PCD 31",
      [L(f"FST_Rotor_{w}_{i}") for i in (1, 2, 3, 4) for w in "AB"], -130, -24, margin=1.25,
      context=L("Small_Drum"), stagger=0.12, move_time=1.5)
    S(10, "Frame posts", "5× printed posts on the plate", [L("Post")], -125, -14, margin=0.95,
      context=plate, stagger=0.1)
    S(11, "Drum A bearings into the plate", "2× 6806 (30 × 42 × 7): one from the back, one from the front, 21.5 mm apart",
      [L("REF_Bearing_A_30x42_rear"), L("REF_Bearing_A_30x42_front")], -150, -14, margin=1.1,
      stagger=0.9)
    show_part(F, count, L("Big_Drum_A"), "Drum A", "One printed part: 150° sector, Ø30 shaft at the back, output tube at the front", 30)
    S(16, "Drum A: tensioner sliders", "A printed slider drops into the pocket at each end of the drum",
      [L("Rope_Tensioner_Slider_2"), L("Rope_Tensioner_Slider_1")], -150, -22, margin=1.15,
      context=L("Big_Drum_A"), stagger=0.6)
    S(17, "Drum A: tensioner nuts", "M3 hex nut into the slot in each slider",
      [L("REF_Tensioner_A_plus_nut"), L("REF_Tensioner_A_minus_nut")], -150, -22, margin=1.15,
      context=L("Big_Drum_A"), stagger=0.6)
    S(18, "Drum A: M5 heat-set insert", "Into the end of the shaft, for the rear cap",
      [L("FST_Insert_A_shaft")], 150, -12, margin=2.0, ghost=L("Big_Drum_A"))
    S(19, "Drum A into the plate", "Its Ø30 shaft slides through both bearings; the shoulder stops on the front one",
      [drum_a], -125, -14, margin=1.05, context=plate)
    S(20, "Rear cap", "Printed cap over the rear bearing's inner race", [L("A_Shaft_Cap")], 150, -12,
      margin=2.2, context=L("REF_Bearing_A_30x42_rear"))
    S(20, "Rear cap screw", "M5 × 12 socket head into the shaft's insert", [L("FST_Screw_A_cap")], 150, -12,
      margin=2.2, context=L("A_Shaft_Cap"), move_time=1.6)
    show_part(F, count, L("Big_Drum_B"), "Drum B", "One printed part: 150° sector, hub and Ø40 journal for the front wall bearing", -150)
    S(21, "Drum B: bearings", "2× 6806 (30 × 42 × 7) from the rear: the front one first, a 4 mm gap between",
      [L("REF_Bearing_30x42_B2"), L("REF_Bearing_30x42_B1")], -140, -15, margin=1.2, ghost=L("Big_Drum_B"), stagger=0.6)
    S(22, "Drum B: M3 heat-set inserts", "6× into the end of the Ø40 journal, PCD 32",
      [L("FST_Insert_B_hub")], -140, -15, margin=1.2, ghost=L("Big_Drum_B"), stagger=0.12, move_time=1.0)
    S(23, "Drum B: tensioner sliders and nuts", "Same as drum A",
      [L("Rope_Tensioner_Slider_4", "REF_Tensioner_B_plus_nut"), L("Rope_Tensioner_Slider_3", "REF_Tensioner_B_minus_nut")],
      -140, -15, margin=1.2, context=L("Big_Drum_B"), stagger=0.6)
    S(24, "Drum B over A's tube", "Both 6806 slide onto A's Ø30 tube; A's Ø22 nose passes through B's journal", [drum_b], -125, -14,
      margin=1.0, context=L("Big_Drum_A"))

    behind = L("Motor_Plate", "REF_Motor", "REF_moteus", "FST_Stator", "FST_Insert_moteus", "FST_Standoff",
               "FST_Screw_moteus", "Post", "Small_Drum", "FST_Rotor", "A_Shaft_Cap", "FST_Screw_A_cap",
               "REF_Bearing_A_30x42", "FST_Nut_posts")
    group = {"A": drum_a, "B": drum_b}
    for stage, look, az in (("A", np.array([-40, 50, -12]), -165), ("B", np.array([40, 79, -12]), -15)):
        n = next(count)
        st = F.t
        if stage == "A":                                          # drum B sits in front of A's rope plane
            F.ghost.append((drum_b, st, st + 4.9, 0.15))
        F.cam(st + 0.7, look, 240, az, -12)
        F.rope.append((stage, st + 0.9, st + 4.6))
        F.cam(st + 4.7, look, 230, az + (10 if stage == "A" else -10), -16)
        F.captions.append((st + 0.15, st + 4.7, str(n), f"Rope {stage}",
                           "In through the rim hole, through the slider, knot in its pocket; ~5 turns on the small drum; same at the other end"))
        F.t = st + 4.8
        for j, s_ in enumerate((1, -1)):                          # one close-up per tensioner, onto its open pocket
            screw = L(f"REF_Tensioner_{stage}_{'plus' if s_ > 0 else 'minus'}_M3x14_screw")
            nut_ = L(f"REF_Tensioner_{stage}_{'plus' if s_ > 0 else 'minus'}_nut")
            slider = [SLIDER[(stage, s_)]]
            o, xd, zd = ts_frame(stage, s_)
            f = UP_R @ (zd + 0.4 * xd)
            f /= np.linalg.norm(f)
            az_, el_ = math.degrees(math.atan2(f[1], f[0])), math.degrees(math.asin(f[2]))
            t0 = F.t
            hide = []
            if s_ < 0:                                            # pocket opens to the rear: see past the plate
                hide += behind + (drum_a if stage == "B" else [])
            elif stage == "A":
                hide += drum_b
            S(0 if j == 0 else -1, f"Tension rope {stage} ({j + 1}/2): screw",
              "M3 × 12 socket head in from the drum's end face, along the rope line, into the nut in the slider",
              [screw], az_, el_, margin=1.3, context=slider + nut_, lead=0.6, move_time=1.8)
            S(-1, f"Tension rope {stage} ({j + 1}/2): tighten",
              "Turning the screw (2.5 mm hex key) pulls the slider, and the rope knotted in it, toward the drum's end",
              [screw + slider + nut_], az_, el_, margin=1.5, lead=0.4, move_time=2.6, hold=1.0)
            F.ghost.append((hide, t0, F.t, 0.1))
            F.ghost.append((L(f"Big_Drum_{stage}"), t0, F.t, 0.4))   # see the screw, nut and rope line inside
            F.highlight.append((slider + nut_ + screw, t0 + 3.0, F.t - 0.3))
            if s_ < 0:
                F.floor_ghost.append((t0, F.t))
    S(29, "Front wall: tip bearings", "6902 into each pocket", [L("REF_Bearing_15x28_tip")], -140, -14,
      margin=1.1, ghost=L("Front_Wall"), stagger=0.2)
    S(29, "Front wall: drum B bearing", "6808 (40 × 52 × 7) pressed in from the rear, against the front lip",
      [L("REF_Bearing_40x52_wall")], -140, -14, margin=1.1, ghost=L("Front_Wall"))
    S(30, "Front wall", "The 6808 slides onto drum B's journal; tip bearings seat on the small drums",
      [L("Front_Wall", "REF_Bearing_15x28_tip", "REF_Bearing_40x52_wall")], -128, -12, margin=0.95, context=L("Big_Drum_B"))
    S(31, "Frame bolts", "5× M4 × 110 socket head through wall, posts and plate",
      [L(f"FST_Bolt_posts_{i}") for i in range(1, 6)], -135, -10, margin=1.0, context=L("Front_Wall"),
      stagger=0.15, move_time=1.8)
    S(32, "Frame nuts", "5× M4 hex nut on the plate back", [L(f"FST_Nut_posts_{i}") for i in range(1, 6)],
      rear, -10, margin=1.0, context=plate, stagger=0.15, move_time=1.3)
    S(33, "Flange B: captured nuts", "6× M3 hex nut into the rear pockets, PCD 70", [L("FST_Nut_B_flange")],
      -150, -8, margin=1.4, ghost=L("Output_Flange_B"), stagger=0.1, move_time=1.0)
    S(34, "Output flange B", "Over A's nose; its hub clamps the 6808's inner race to drum B (only after the wall)",
      [L("Output_Flange_B", "FST_Nut_B_flange")], -140, -10, margin=1.15, context=L("Big_Drum_B"))
    S(35, "Flange B screws", "6× M3 × 16 socket head into the journal inserts, PCD 32", [L("FST_Screw_B_flange")],
      -150, -10, margin=1.2, context=L("Output_Flange_B"), stagger=0.12, move_time=1.4)
    S(36, "Test arm B", "Onto flange B, located by its six screws", [L("Test_Arm_B")], -130, -10, margin=1.0,
      context=L("Output_Flange_B"))
    S(37, "Arm B screws", "6× M3 × 10 socket head into the flange nuts, PCD 70", [L("FST_Screw_B_arm")],
      -145, -10, margin=1.15, context=L("Output_Flange_B"), stagger=0.12, move_time=1.4)
    S(38, "Drum A nose: M3 heat-set inserts", "4× into the end of A's Ø22 nose, PCD 14", [L("FST_Insert_A_nose")],
      -150, -6, margin=2.6, stagger=0.15, move_time=1.0)
    S(38, "Flange A: M3 heat-set inserts", "6× into the front face, PCD 48", [L("FST_Insert_A_flange")],
      -150, -6, margin=1.5, ghost=L("Output_Flange_A"), stagger=0.1, move_time=1.0)
    S(39, "Output flange A", "Onto the end of A's Ø22 nose", [L("Output_Flange_A", "FST_Insert_A_flange")],
      -150, -6, margin=1.3, context=L("Output_Flange_B"))
    S(40, "Flange A screws", "4× M3 × 10 socket head into the nose inserts, PCD 14", [L("FST_Screw_A_flange")],
      -155, -6, margin=1.4, context=L("Output_Flange_A"), stagger=0.12, move_time=1.4)
    S(41, "Test arm A", "Onto A's spigot; M5 load holes at 50–150 mm", [L("Test_Arm_A")], -130, -10,
      margin=0.95, context=L("Output_Flange_B"))
    S(42, "Arm A screws", "6× M3 × 10 socket head into the flange inserts, PCD 48", [L("FST_Screw_A_arm")],
      -145, -8, margin=1.3, context=L("Output_Flange_A"), stagger=0.12, move_time=1.4)
    S(43, "Rib inserts", "4× M4 heat-set inserts, two into each plate's rib (motor plate and front wall)",
      [L("FST_Insert_ribs")], -125, 24, margin=1.0, context=plate + L("Front_Wall"), stagger=0.25, move_time=1.0)
    S(44, "Base mount", "Slides under the ribs; the rig stands on it, arms up", [L("Base_Mount")], -118, -18,
      margin=0.9, context=plate + L("Front_Wall"))
    S(45, "Base mount screws", "4× M4 × 12 socket head from underneath into the rib inserts", [L("FST_Screw_base")],
      -118, -30, margin=0.9, context=L("Base_Mount"), stagger=0.15, move_time=1.4)
    F.floor_ghost.append((F.t - 13.0, F.t + 0.5))               # see the parts that go in from below
    F.desk = (F.t - 0.2, F.t + 0.8)
    drivetrain(F, L, F.t)
    F.cards[-1] = (F.cards[-1][0], F.cards[-1][1], "Coaxial Capstan v3", f"{next(count) - 1} steps  ·  CAD animation, threads omitted")
    return F


def rim_point(stage, theta):
    c = DIR_A if stage == "A" else DIR_B
    a = math.radians(c + (35 if stage == "A" else -35)) + theta
    return np.array([OUT[0] + BIG_R * math.cos(a), A_MID if stage == "A" else B_MID, OUT[1] + BIG_R * math.sin(a)])


def arm_point(stage, theta, r):
    a = math.radians(ARM_DIR[stage]) + theta
    return np.array([OUT[0] + r * math.cos(a), ARM_Y[stage], OUT[1] + r * math.sin(a)])


def rope_point(stage, theta, frac):
    pts = rope_path(MA if stage == "A" else MB, A_MID if stage == "A" else B_MID, theta)
    return pts[int(frac * 41)]


def author_exploded(meta, verts):
    """Short loop: assembled -> labelled exploded orbit -> assembled."""
    G = Film(meta, verts)
    L = lambda *n: labels_of(meta, *n)
    set_offsets(G, L)
    G.base = set(L("Motor_Plate", "REF_Motor", "REF_moteus", "FST_Stator", "FST_Insert_moteus", "FST_Screw_moteus", "FST_Standoff_moteus"))
    G.explode = (1.5, 4.5)
    G.implode = (14.0, 17.0)
    G.cam(0.0, *HERO)
    G.cam(1.5, HERO[0], 880, -115, -16)
    G.cam(4.5, EXPLODED_CAM[0], EXPLODED_CAM[1], -32, -12)
    G.cam(14.0, EXPLODED_CAM[0], EXPLODED_CAM[1], -8, -10)
    G.cam(17.0, HERO[0], 880, -95, -16)
    G.cam(18.5, HERO[0], 880, -85, -16)
    G.labels.append((4.8, 13.8, explode_labels(L)))
    G.captions.append((4.6, 13.8, "", "Exploded view", "Front to back in insertion order"))
    G.rope_fade = (1.0, 1.5)
    G.rope_return = (17.0, 17.5)
    G.duration = 18.5
    return G


# ---------------------------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# Five-bar leg (mirrors generate.leg_*): thighs on the two outputs, shins meet at the foot
# ---------------------------------------------------------------------------------------------
LEG_L1, LEG_L2, FOOT_R = 100.0, 200.0, 12.0
LEG_Q0 = {"A": ARM_DIR["A"] - 60.0, "B": ARM_DIR["B"]}          # thigh angles (deg) with the outputs centred


def leg_ik(r, phi):
    lo, hi = 0.0, 179.0
    for _ in range(60):
        mid = (lo + hi) / 2
        sp = math.radians(mid)
        rr = LEG_L1 * math.cos(sp) + math.sqrt(LEG_L2 ** 2 - (LEG_L1 * math.sin(sp)) ** 2)
        lo, hi = (mid, hi) if rr > r else (lo, mid)
    return phi - lo, phi + lo


def five_bar(qa, qb):
    """Knees and foot (model x, z) for thigh angles qa, qb (deg)."""
    k = {w: np.array([OUT[0] + LEG_L1 * math.cos(math.radians(q)), OUT[1] + LEG_L1 * math.sin(math.radians(q))])
         for w, q in (("A", qa), ("B", qb))}
    m, d = (k["A"] + k["B"]) / 2, k["B"] - k["A"]
    half = np.linalg.norm(d) / 2
    n = np.array([-d[1], d[0]]) / (2 * half)
    if np.linalg.norm(m + n - np.array(OUT)) < np.linalg.norm(m - n - np.array(OUT)):
        n = -n
    return k, m + math.sqrt(LEG_L2 ** 2 - half ** 2) * n


LEG_K0, LEG_F0 = five_bar(LEG_Q0["A"], LEG_Q0["B"])
LEG_R0 = float(np.linalg.norm(LEG_F0 - np.array(OUT)))


# 4-bar leg (mirrors generate.leg4_*): B swings the thigh, A's crank sets the shin via a parallel coupler
L4_LEN = float(os.environ.get("CAPSTAN_LEG_L", "200"))             # as generate.L4_LEN
L4_K = L4_LEN / 150.0
L4_L1, L4_L2, L4_C, L4_DELTA, L4_FOOT_R = L4_LEN, L4_LEN, 55.0, -20.0, 23.0
L4_Q0 = {"thigh": -60.0, "shin": -120.0}


def leg4_points(qt, qs):
    u = lambda q: np.array([math.cos(math.radians(q)), math.sin(math.radians(q))])
    h = np.array(OUT)
    k = h + L4_L1 * u(qt)
    ql = qs + 180 + L4_DELTA                                    # shin lever = crank direction
    return k, k + L4_L2 * u(qs), h + L4_C * u(ql), k + L4_C * u(ql)


def leg4_ik(r, phi):
    cos_k = (r ** 2 - L4_L1 ** 2 - L4_L2 ** 2) / (2 * L4_L1 * L4_L2)
    kb = math.degrees(math.acos(max(-1.0, min(1.0, cos_k))))
    a = math.degrees(math.atan2(L4_L2 * math.sin(math.radians(kb)), L4_L1 + L4_L2 * math.cos(math.radians(kb))))
    return phi + a, phi + a - kb


L4_K0, L4_F0, L4_C0, L4_D0 = leg4_points(L4_Q0["thigh"], L4_Q0["shin"])
L4_R0 = float(np.linalg.norm(L4_F0 - np.array(OUT)))

# 3-DOF hip ab/ad (mirrors generate.ab_*): drum B's stage mapped onto an axis along X
AB_Y, AB_Z, AB_X0, B_Y0 = 45.0, -60.0, -142.0, 70.0     # v5: ab/ad axis almost level with the hip
AB_AL = math.radians(DIR_B + 90.0)
AB_C = np.array([0.0, AB_Y, AB_Z])
AB_RAIL_X = -40.0                                       # v6: the rail sits behind the middle of the cradle
ABAD_SPIN = ("Abad_Motor_rotor", "Abad_Motor_magnet_holder", "Abad_Motor_magnet_disc", "Abad_Small_Drum",
             "Abad_HW_small_drum_screws")


def ab_map(pts):
    """Main-frame points of drum B's stage -> their place on the ab/ad capstan (main +Y -> +X)."""
    pts = np.atleast_2d(pts)
    dx, dz, d = pts[:, 0] - OUT[0], pts[:, 2] - OUT[1], pts[:, 1] - B_Y0
    c, s_ = math.cos(AB_AL), math.sin(AB_AL)
    return np.stack([AB_X0 + d, AB_Y + dx * c + dz * s_, AB_Z + dx * s_ - dz * c], axis=1)


def rot_x(phi):
    c, s_ = math.cos(phi), math.sin(phi)
    return np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])                  # y toward z


LEGS = {
    "fivebar": dict(r0=LEG_R0, foot=FOOT_R, crouch=190.0, push=282.0, tuck=245.0, land=195.0,
                    squat=lambda u: LEG_R0 - 42.0 * math.sin(2 * math.pi * 2 * u), step=(22.0, 16.0),
                    title="five-bar leg", own="Leg_", other="Leg4_",
                    squat_note="Outputs turn opposite ways: the knees spread and the foot comes up",
                    step_note="Outputs turn together to swing the leg, opposite ways to lift the foot"),
    "4bar": dict(r0=L4_R0, foot=L4_FOOT_R, crouch=200.0 * L4_K, push=290.0 * L4_K, tuck=255.0 * L4_K, land=205.0 * L4_K,
                 squat=lambda u: L4_R0 - 95.0 * L4_K * (1 - math.cos(2 * math.pi * 2 * u)) / 2, step=(25.0, 15.0),
                 title="4-bar leg", own="Leg4_", other="Leg_",
                 squat_note="B swings the thigh; A's crank folds the knee through the parallel coupler rod",
                 step_note="Thigh and knee driven separately: hip swing on B, knee on A"),
}


def hop_track(leg, slow=0.4):
    """Real-time hop, played at `slow` speed: list of (video dt, r, phi, hip height above the floor)."""
    g, dt, out = 9810.0, 1 / 600, []
    fr, r0 = leg["foot"], leg["r0"]
    def seg(T, rf):
        n = max(1, int(T / dt))
        for i in range(n):
            r = rf((i + 1) / n)
            out.append((dt / slow, r, -90.0, r + fr))
    c, p, tk, ld = leg["crouch"], leg["push"], leg["tuck"], leg["land"]
    seg(0.30, lambda u: r0 - (r0 - c) * ease(u))                                          # crouch
    seg(0.10, lambda u: c + (p - c) * (1 - math.cos(math.pi / 2 * u)))                    # push
    v0 = (p - c) * (math.pi / 2) / 0.10
    z0 = p + fr
    tau = 0.0
    while True:                                                                           # flight, tucked
        tau += dt
        z = z0 + v0 * tau - g / 2 * tau ** 2
        r = p - (p - tk) * ease(min(1.0, tau / 0.15))
        if z - fr <= r and tau > 0.05:
            break
        out.append((dt / slow, r, -90.0, z, "air"))
    r_land = r
    seg(0.12, lambda u: r_land - (r_land - ld) * math.sin(math.pi / 2 * u))             # land
    seg(0.35, lambda u: ld + (r0 - ld) * ease(u))                                         # stand back up
    seg(0.25, lambda u: r0)
    return out


def leg_film(meta, verts, part="full", variant="fivebar"):
    """The rig on the extrusion stand as a one-leg hopper: squat, step in the air, hop.
    variant "3dof": the 4-bar leg on the ab/ad hip (motor C), with sideways swings in the air."""
    three = variant == "3dof"
    if three:
        variant = "4bar"
    leg = LEGS[variant]
    F = Film(meta, verts)
    F.theme = "dark"
    L = lambda *n: labels_of(meta, *n)
    F.hidden = set(L("Test_Arm", "Base_Mount", leg["other"], "Abad_", "Hip_"))   # 2-DOF: no ab/ad unit
    if three:                                                     # hip frame replaces the plain rail carriage
        F.hidden = set(L("Test_Arm", "Base_Mount", leg["other"], "Rail_Carriage", "REF_MGN12H", "Carriage_HW"))
    F.base = set(meta)
    F.world_cam = True
    r0, fr = leg["r0"], leg["foot"]
    track = []                                                   # (video dt, r, phi, hip height)
    def hold(T):
        for _ in range(int(T * 120)):
            track.append((1 / 120, r0, -90.0, r0 + fr))
    marks = {}
    if part == "full":
        hold(3.4)
        marks["squat"] = len(track)
        for i in range(int(8.0 * 120)):                           # two slow squats, foot on the floor
            r = leg["squat"]((i + 1) / (8.0 * 120))
            track.append((1 / 120, r, -90.0, r + fr))
        marks["lift"] = len(track)
        for i in range(int(1.4 * 120)):                           # carriage lifts the hip 70 mm
            track.append((1 / 120, r0, -90.0, r0 + fr + 70.0 * ease((i + 1) / (1.4 * 120))))
        marks["step"] = len(track)
        dr, dphi = leg["step"]
        for i in range(int(7.5 * 120)):                           # stepping in the air: foot traces a loop
            u = (i + 1) / (7.5 * 120)
            env = math.sin(math.pi * u) ** 0.5
            track.append((1 / 120, r0 + dr * env * math.cos(2 * math.pi * 3 * u),
                          -90.0 + dphi * env * math.sin(2 * math.pi * 3 * u), r0 + fr + 70.0))
        for i in range(int(1.4 * 120)):
            track.append((1 / 120, r0, -90.0, r0 + fr + 70.0 * (1 - ease((i + 1) / (1.4 * 120)))))
        hold(0.6)
    marks["hop"] = len(track)
    hop_marks = []
    for _ in range(3 if part == "full" else 2):
        hop_marks.append(len(track))
        track += hop_track(leg)
    marks["end"] = len(track)
    hold(2.8 if part == "full" else 0.0)
    tt = np.cumsum([0.0] + [x[0] for x in track])[:-1]
    rr = np.array([x[1] for x in track]); pp = np.array([x[2] for x in track]); hh = np.array([x[3] for x in track])
    ab = np.zeros(len(track))                                     # ab/ad (deg): 0 whenever the foot is down
    if three:
        if part == "full":
            i0, i1 = marks["step"], marks["step"] + int(7.5 * 120)
            u = np.linspace(0, 1, i1 - i0)
            ab[i0:i1] = 18.0 * np.sin(np.pi * u) ** 0.5 * np.sin(2 * np.pi * 1.5 * u)
        air = np.array([len(x) > 4 for x in track])
        for n, h0 in enumerate(hop_marks):                        # tilt a little in each flight, back to 0 to land
            idx = [i for i in range(h0, (hop_marks + [len(track)])[n + 1]) if air[i]]
            if idx:
                u = np.linspace(0, 1, len(idx))
                ab[idx] = (8.0 if n % 2 == 0 else -8.0) * np.sin(np.pi * u)
    at = lambda arr, t: float(np.interp(t, tt, arr))
    F.duration = float(tt[-1])
    F.up_at = lambda t: (np.eye(3), np.array([0.0, 0.0, at(hh, t) - OUT[1]]))   # hanging: hip at its height above the floor

    def turn(label, angle, pivot0, pivot):
        """Pose: rotate by angle (rad, XZ) about pivot0, then move pivot0 to pivot."""
        dp, q = rot_xz(angle, np.array([pivot0[0], 0.0, pivot0[1]]))
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        return R.reshape(3, 3), dp + np.array([pivot[0] - pivot0[0], 0.0, pivot[1] - pivot0[1]])

    if variant == "fivebar":
        pose = lambda t: leg_ik(at(rr, t), at(pp, t))
        F.angle_fn = lambda t: {"A": math.radians(pose(t)[0] - LEG_Q0["A"]), "B": math.radians(pose(t)[1] - LEG_Q0["B"])}
        def shin(w):
            def f(t):
                k, ft = five_bar(*pose(t))
                a0 = math.atan2(LEG_F0[1] - LEG_K0[w][1], LEG_F0[0] - LEG_K0[w][0])
                a1 = math.atan2(ft[1] - k[w][1], ft[0] - k[w][0])
                return turn(w, a1 - a0, LEG_K0[w], k[w])
            return f
        for w in "AB":
            F.custom[f"Leg_Shin_{w}"] = shin(w)
        F.custom["Leg_Foot"] = lambda t: (np.eye(3), np.array([five_bar(*pose(t))[1][0] - LEG_F0[0], 0.0,
                                                               five_bar(*pose(t))[1][1] - LEG_F0[1]]))
        def joint5(j):                                            # pins and bearings ride with their joint
            def f(t):
                k, ft = five_bar(*pose(t))
                p, p0 = (k["A"], LEG_K0["A"]) if j == "knee_A" else (k["B"], LEG_K0["B"]) if j == "knee_B" else (ft, LEG_F0)
                return np.eye(3), np.array([p[0] - p0[0], 0.0, p[1] - p0[1]])
            return f
        for k in L("Leg_HW_"):
            F.custom[k] = joint5(k[len("Leg_HW_"):].rsplit("_", 1)[0])
    else:
        pose = lambda t: leg4_ik(at(rr, t), at(pp, t))          # (thigh, shin) angles
        F.angle_fn = lambda t: {"A": math.radians(pose(t)[1] - L4_Q0["shin"]), "B": math.radians(pose(t)[0] - L4_Q0["thigh"]),
                                "C": math.radians(at(ab, t))}
        if three:
            F.abad_fn = lambda t: math.radians(at(ab, t))
        def shin4(t):
            qt, qs = pose(t)
            return turn("s", math.radians(qs - L4_Q0["shin"]), L4_K0, leg4_points(qt, qs)[0])
        def coupler4(t):
            qt, qs = pose(t)
            return turn("c", math.radians(qt - L4_Q0["thigh"]), L4_C0, leg4_points(qt, qs)[2])
        F.custom["Leg4_Shin"] = shin4
        F.custom["Leg4_Foot"] = shin4                              # the bulb rides on the shin's claw
        for k in L("Leg4_Shin_Plate"):                             # v6 sandwich plates ride on the shin
            F.custom[k] = shin4
        F.custom["Leg4_Coupler"] = coupler4
        def joint4(j):
            idx = {"knee": 0, "crank": 2, "lever": 3}[j]
            p0 = (L4_K0, L4_F0, L4_C0, L4_D0)[idx]
            def f(t):
                p = leg4_points(*pose(t))[idx]
                return np.eye(3), np.array([p[0] - p0[0], 0.0, p[1] - p0[1]])
            return f
        for k in L("Leg4_HW_"):
            j = meta[k].get("part", k)[len("Leg4_HW_"):].rsplit("_", 1)[0]     # split hardware: its family
            F.custom[k] = shin4 if j in ("foot", "shin") else joint4(j)
    sx = AB_RAIL_X if three else 0.0                              # 3-DOF: the rail sits behind the ab/ad unit
    sdy = (-85.0 - -137.0) if three else 0.0                       # v6: spine 85 mm behind the module
    variant_meshes = any("group" in v for v in meta.values())
    t_rest = hold_h = r0 + fr - OUT[1]
    for k in L("Stand_"):                                         # the stand stays put in the world
        if variant_meshes:                                        # v5: stand meshes in model coordinates at rest
            F.custom[k] = lambda t: (np.eye(3), np.array([0.0, 0.0, t_rest]) - np.asarray(UP_T, float))
        else:
            F.custom[k] = lambda t: (np.eye(3), np.array([sx, sdy, 0.0]) - np.asarray(UP_T, float))
    F.rope = [("A", -2.0, -1.0), ("B", -2.0, -1.0)]
    F.beads.append((0.0, F.duration))
    F.readouts.append((0.0, F.duration, ["A", "B", "C"] if three else ["A", "B"]))
    T = lambda key: float(tt[marks[key]]) if marks[key] < len(tt) else F.duration
    side, front = (-128, -6), (-90, -4)
    look = np.array([0.0, 40.0, 300.0])
    if three and part == "full":
        hip_look = np.array([-120.0, 20.0, r0 + fr - OUT[1] + 20.0])  # ab/ad unit: motor C, moteus C, drum
        F.cam(0.0, hip_look, 700, 35, -16)
        F.cam(3.2, hip_look, 700, -40, -10)
        F.cards.append((0.2, 3.2, "Capstan leg, 3 DOF", "Motor C adds hip ab/ad through a third 8:1 capstan at the back"))
        F.cam(T("squat") + 1.2, look + [0, 0, 120], 1500, -128, -6)
        F.captions.append((T("squat") + 0.2, T("lift") - 0.2, "", "Squat", leg["squat_note"]))
        F.cam(T("lift"), look + [0, 0, 120], 1500, -128, -6)
        F.cam(T("step") + 0.8, look + [0, 0, 120], 1350, -20, -6)
        F.captions.append((T("step") + 0.2, T("hop") - 0.6, "", "Hip ab/ad",
                           "Motor C turns the whole module about the body axis, up to ±18°, while the leg steps"))
        F.cam(T("hop") - 0.6, look + [0, 0, 100], 1450, -55, -4)
        F.captions.append((T("hop") + 0.2, T("end"), "", "Jumping (shown at 0.4× speed)",
                           "Crouch, push, tilt a few degrees in the air, land with ab/ad back at zero"))
        F.cam(F.duration, look + [0, 0, 100], 1450, -100, -4)
        F.cards.append((T("end") + 0.2, F.duration, "Capstan leg, 3 DOF", "CAD animation  ·  kinematics only, not a dynamics simulation"))
    elif three:
        F.cam(0.0, look + [0, 0, 100], 1400, -60, -5)
    elif part == "full":
        leg_look = np.array([0.0, 110.0, (r0 + fr) / 2 + 40.0])       # close on the leg itself
        F.cam(0.0, leg_look, 760, -105, -4)
        F.cam(2.4, leg_look, 760, -62, -8)
        F.cam(T("squat") + 1.2, look, 1250, *side)
        F.cards.append((0.2, 2.4, f"Capstan {leg['title']}", "Rig on an MGN12 rail on a 2040 extrusion stand"))
        F.captions.append((T("squat") + 0.2, T("lift") - 0.2, "", "Squat", leg["squat_note"]))
        F.cam(T("lift"), look, 1250, *side)
        F.cam(T("step") + 0.5, look + [0, 0, 40], 1050, *front)
        F.captions.append((T("step") + 0.2, T("hop") - 0.6, "", "Stepping in the air", leg["step_note"]))
        F.cam(T("hop") - 0.6, look + [0, 0, 40], 1250, -125, -3)
        F.captions.append((T("hop") + 0.2, T("end"), "", "Hopping (shown at 0.4× speed)", "Crouch, push, tuck, land  ·  illustrative take-off speed"))
        F.cam(F.duration, look + [0, 0, 40], 1250, -105, -3)
        F.cards.append((T("end") + 0.2, F.duration, f"Capstan {leg['title']}", "CAD animation  ·  kinematics only, not a dynamics simulation"))
    else:
        F.cam(0.0, look + [0, 0, 40], 1250, -115, -4)
    return F


def author_v4_build(meta, verts):
    """v4 Stanley-style leg: the physical build order, hanging as on the rail, one sub-assembly per step."""
    F = Film(meta, verts)
    F.theme = "dark"
    L = lambda *n: labels_of(meta, *n)
    F.hidden = set(L("Test_Arm", "Base_Mount", "Leg_", "Rail_Carriage", "REF_MGN12H", "Carriage_HW", "Stand_"))
    plate = L("Motor_Plate")
    F.base = set(plate)
    hip_floor = L4_R0 + L4_FOOT_R
    F.up_at = lambda t: (np.eye(3), np.array([0.0, 0.0, hip_floor - OUT[1]]))
    F.abad_fn = lambda t: 0.0
    y = lambda dy: (0, dy, 0)
    motor = lambda w: L(f"REF_Motor_{w}_")
    fst = lambda *n: L(*[f"FST_{x}" for x in n])
    groups = {
        "motors": motor("A") + motor("B") + fst("Stator"), "moteus": L("REF_moteus") + fst("Insert_moteus", "Standoff_moteus", "Screw_moteus"),
        "small": L("Small_Drum") + fst("Rotor"), "brgA": L("REF_Bearing_A_30x42"),
        "drumA": L("Big_Drum_A", "Rope_Tensioner_Slider_1", "Rope_Tensioner_Slider_2", "REF_Tensioner_A", "FST_Insert_A_shaft"),
        "cap": L("A_Shaft_Cap", "FST_Screw_A_cap"),
        "drumB": L("Big_Drum_B", "REF_Bearing_30x42", "Rope_Tensioner_Slider_3", "Rope_Tensioner_Slider_4", "REF_Tensioner_B", "FST_Insert_B_hub"),
        "posts": L("Post_"), "wall": L("Front_Wall", "REF_Bearing_15x28_tip", "REF_Bearing_40x52_wall", "FST_Insert_gantry"),
        "bracket": L("End_Bracket"), "bolts": fst("Bolt_posts", "Nut_posts"),
        "flB": L("Output_Flange_B") + fst("Nut_B_flange", "Screw_B_flange"), "thigh": L("Leg4_Thigh") + fst("Screw_B_arm"),
        "flA": L("Output_Flange_A") + fst("Insert_A_nose", "Screw_A_flange", "Insert_A_flange"), "crank": L("Leg4_Crank") + fst("Screw_A_arm"),
        "lower": L("Leg4_Shin", "Leg4_Coupler", "Leg4_Foot", "Leg4_HW_"),
        "gposts": L("Gantry_Post"), "gantry": L("Gantry", "REF_Bearing_17x30_gantry") + fst("Bolt_gantry"),
        "abad": L("Abad_", "Hip_"),
    }
    for g_ in ("gposts",):
        groups["gantry"] = [k for k in groups["gantry"] if k not in groups[g_]]
    F.offset(groups["motors"], y(140))
    F.offset(groups["moteus"], y(-70))
    F.offset(groups["small"], y(160))
    F.offset(L("REF_Bearing_A_30x42_rear"), y(-60))
    F.offset(L("REF_Bearing_A_30x42_front"), y(80))
    F.offset(groups["drumA"], y(220))
    F.offset(groups["cap"], y(-60), turns=4)
    F.offset(groups["drumB"], y(240))
    F.offset(groups["posts"], y(140))
    F.offset(groups["wall"], y(160))
    F.offset(groups["bracket"], (-140, 0, 0))
    F.offset(groups["bolts"], y(70))
    F.offset(groups["flB"], y(90))
    F.offset(groups["thigh"], y(110))
    F.offset(groups["flA"], y(110))
    F.offset(groups["crank"], y(120))
    F.offset(groups["lower"], y(130))
    F.offset(groups["gposts"], y(110))
    F.offset(groups["gantry"], y(130))
    F.offset(groups["abad"], (-220, 0, 0))
    F.explode = (1000.0, 1001.0)                                # the opening shows it assembled
    F.hide_at = 3.2
    F.cam(0.0, np.array([-40.0, 60.0, -60.0]), 900, -128, -14)
    F.cam(3.0, np.array([-40.0, 60.0, -60.0]), 860, -108, -14)
    F.cards.append((0.2, 3.0, "Stanley-style leg v4", "3 DOF  ·  compact frame  ·  output held on both sides of the leg"))
    F.t = 3.4
    count = iter(range(1, 100))
    S = lambda *a, **k: F.step(next(count), *a, **k)
    S("Motors onto the motor plate", "2× EaglePower 8308, 4× M4 × 12 countersunk each from the back",
      [motor("A") + fst("Stator_A"), motor("B") + fst("Stator_B")], -130, -12, margin=1.0, context=plate, stagger=0.6)
    S("moteus boards", "Connectors outboard; inserts, 2 standoffs and 4× M2.5 × 6 each", [groups["moteus"]], 50, -10, margin=1.1)
    S("Small drums", "4× M3 × 10 into each rotor", [groups["small"]], -130, -15, margin=1.1, context=plate)
    S("Drum A bearings", "2× 6806 into the plate's housing, one from each side", [L("REF_Bearing_A_30x42_rear"), L("REF_Bearing_A_30x42_front")],
      -140, -12, margin=1.4, stagger=0.6)
    S("Drum A", "Shaft through both bearings; sliders and nuts already in", [groups["drumA"]], -130, -12, margin=1.0, context=plate)
    S("Rear cap", "M5 × 12 into the shaft's insert, from the back", [groups["cap"]], 50, -10, margin=2.0)
    S("Drum B", "2× 6806 inside; slides over A's tube", [groups["drumB"]], -130, -12, margin=1.0, context=L("Big_Drum_A"))
    st = F.t
    F.rope += [("A", st + 0.5, st + 3.5), ("B", st + 3.8, st + 6.8)]
    F.cam(st + 0.6, np.array([-30.0, 60.0, -40.0]), 520, -40, -10)
    F.cam(st + 6.8, np.array([30.0, 75.0, -40.0]), 520, -140, -10)
    F.captions.append((st + 0.2, st + 7.0, str(next(count)), "Ropes", "Crossed loops, knotted in the sliders; tension with the M3 screws at the drum ends"))
    F.t = st + 7.2
    S("Posts", "4 corner posts, clear of both drums", [groups["posts"]], -128, -12, margin=1.0, context=plate)
    S("Front wall", "Tip bearings, the 6808 for drum B and the gantry's M4 inserts", [groups["wall"]], -128, -12, margin=1.0)
    S("End bracket", "Slides in from the side; its ears sit on both plates at the left posts", [groups["bracket"]], -150, -10, margin=1.2)
    S("Frame bolts", "4× M4 through wall, posts and plate (through the bracket's ears on the left)", [groups["bolts"]], -128, -10, margin=1.0)
    S("Flange B", "Hub clamps the 6808's inner race; 6× M3 × 16 into B's journal", [groups["flB"]], -140, -10, margin=1.3)
    S("Thigh", "Bolts to flange B, 6× M3 × 10 into the captured nuts", [groups["thigh"]], -130, -10, margin=1.0)
    S("Flange A", "4× M3 × 10 into A's nose", [groups["flA"]], -150, -8, margin=1.6)
    S("Crank", "6× M3 × 10 into flange A's inserts; its stub carries the front bearing", [groups["crank"]], -140, -8, margin=1.3)
    S("Shin, coupler and bulb foot", "M5 pins on MR105ZZ bearings; the soft bulb clips over the shin's claw", [groups["lower"]],
      -125, -8, margin=1.0)
    S("Gantry", "2 posts, then the gantry with its 6902 over the crank's stub: the output is held on both sides",
      [groups["gposts"], groups["gantry"]], -150, -8, margin=1.3, stagger=1.0)
    S("Ab/ad unit", "Motor C, its capstan and the hub on two 6808s; the hub bolts to the end bracket (6× M3 × 16)",
      [groups["abad"]], -40, -10, margin=1.1, move_time=1.8)
    F.abad_rope_t = F.t - 0.5
    st = F.t
    F.cam(st + 1.0, np.array([-60.0, 60.0, -80.0]), 1000, -150, -12)
    F.cam(st + 6.0, np.array([-60.0, 60.0, -80.0]), 1000, -60, -12)
    F.cards.append((st + 1.5, st + 6.0, "Stanley-style leg v4", "Every part above is in print/, step/ and BOM.md"))
    F.duration = st + 6.2
    return F


VARIANT_TITLE = {
    "flange": "2 DOF  ·  test arms  ·  desk", "flange_gantry": "2 DOF  ·  test arms, front gantry  ·  desk",
    "flange_abad": "3 DOF  ·  test arms  ·  desk", "flange_abad_gantry": "3 DOF  ·  test arms, front gantry  ·  desk",
    "leg": "2 DOF  ·  4-bar leg  ·  rail stand", "leg_gantry": "2 DOF  ·  4-bar leg, front gantry  ·  rail stand",
    "leg_abad": "3 DOF  ·  4-bar leg  ·  rail stand", "leg_abad_gantry": "3 DOF  ·  4-bar leg, front gantry  ·  rail stand",
}
DESK_TOP = {False: 62.0 + 8.0, True: 77.0 + 8.0}           # model z of the desk side of the base / desk angle


def author_variant(meta, verts, vname, tris=None):
    """v6: one of the eight builds, in its physical build order and its working orientation (desk builds
    stand on the desk, motors down; leg builds hang on the rail stand), then a short run. Every part goes
    on in its own step; then every heat-set insert, screw, nut and washer that fixes it, one by one, along
    its own axis (screws and nuts turning), with the camera close on the first and pulling back for the rest.
    Sub-assemblies (the shin sandwich, the ab/ad unit, the spine with its blocks) are built off to the side."""
    leg, abad, gantry = "leg" in vname, "abad" in vname, "gantry" in vname
    F = Film(meta, verts)
    F.tris = tris
    F.theme = "dark" if leg else "light"
    L = lambda *n: [k for k in labels_of(meta, *n) if k in meta]
    HW = [k for k in meta if "fit" in meta[k]]
    HWS = set(HW)
    P = lambda *n: [k for k in L(*n) if k not in HWS]                       # parts (and bearings) only
    H = lambda *n: [k for k in HW if meta[k]["part"].startswith(n)]          # hardware of these families
    F.hidden = set()
    plate = L("Motor_Plate")
    F.base = set(plate)
    if leg:
        hip = L4_R0 + L4_FOOT_R
        F.up_at = lambda t: (np.eye(3), np.array([0.0, 0.0, hip - OUT[1]]))
    else:
        top = DESK_TOP[abad]
        F.up_at = lambda t: (np.diag([-1.0, 1.0, -1.0]), np.array([0.0, 0.0, top]))
    R_up = F.up_at(0.0)[0]
    F.abad_fn = (lambda t: 0.0) if abad else None
    y = lambda dy: (0, dy, 0)
    zz = lambda dz: (0, 0, dz)
    motor = lambda w: L(f"REF_Motor_{w}_")

    # --- groups (parts) and the hardware fitted before each group goes on ("pre": on the bench) ---
    front_end = P("End_Bracket_Front")
    G = {
        "motorA": motor("A"), "motorB": motor("B"),
        "moteus": P("REF_moteus"),
        "smallA": P("Small_Drum_A"), "smallB": P("Small_Drum_B"),
        "brgA1": P("REF_Bearing_A_30x42_rear"), "brgA2": P("REF_Bearing_A_30x42_front"),
        "drumA": P("Big_Drum_A", "Rope_Tensioner_Slider_1", "Rope_Tensioner_Slider_2", "REF_Tensioner_A", "FST_Insert_A_shaft"),
        "cap": P("A_Shaft_Cap", "FST_Screw_A_cap"),
        "drumB": P("Big_Drum_B", "REF_Bearing_30x42", "Rope_Tensioner_Slider_3", "Rope_Tensioner_Slider_4", "REF_Tensioner_B"),
        "posts": [k for k in P("Post_", "Top_Rail") if not k.startswith("Gantry_Post")],
        "wall": P("Front_Wall", "REF_Bearing_15x28_tip", "REF_Bearing_40x52_wall"),
        "bracket": [k for k in P("End_Bracket") if k not in front_end], "bracket_f": front_end,
        "trunnion": P("Hip_Trunnion"), "cradle_f": P("Hip_Front_Plate", "Hip_Bearing_front"),
        "flB": P("Output_Flange_B"), "armB": P("Leg4_Thigh", "Test_Arm_B"),
        "flA": P("Output_Flange_A"), "armA": P("Leg4_Crank", "Test_Arm_A"),
        "shin": P("Leg4_Shin", "Leg4_HW_knee_brg", "Leg4_HW_lever_brg"),
        "coupler": P("Leg4_Coupler", "Leg4_HW_crank_brg"), "foot": P("Leg4_Foot"),
        "gposts": P("Gantry_Post"), "gantry": [k for k in P("Gantry", "REF_Bearing_17x30_gantry") if not k.startswith("Gantry_Post")],
        "ab_hub": P("Abad_Drum", "Abad_Slider", "Abad_REF_Tensioner_minus_nut", "Abad_REF_Tensioner_plus_nut", "Abad_Bearings_6808"),
        "ab_inner": P("Hip_Inner_Plate"), "ab_posts": P("Hip_Posts"),
        "ab_outer": P("Hip_Outer_Plate"), "ab_motor": P("Abad_Motor", "Abad_moteus"),
        "ab_small": P("Abad_Small_Drum", "Abad_Bearing_tip"),
        "spine": P("Hip_Spine"), "blocks": P("Hip_REF_MGN12H", "REF_MGN12H_blocks"),
        "mount": P("Desk_Base", "Desk_Angle", "Rail_Carriage"), "stand": P("Stand_"),
    }
    pre = {"posts": H("FST_Insert_posts", "FST_Insert_rails"), "flB": H("FST_Insert_B_flange"),
           "shin": H("Leg4_HW_shin_screws", "Leg4_HW_shin_inserts"), "spine": H("Hip_HW_block_screws"),
           "mount": H("Mount_HW_block_screws") if leg and not abad else [],
           "drumA": [k for k in H("REF_Tensioner_A") if meta[k]["fit"]["kind"] == "nut"],
           "drumB": [k for k in H("REF_Tensioner_B") if meta[k]["fit"]["kind"] == "nut"],
           "ab_hub": [k for k in H("Abad_REF_Tensioner") if meta[k]["fit"]["kind"] == "nut"] + H("Abad_HW_hub_inserts")}
    ab_unit = [k for g_ in ("ab_hub", "ab_inner", "ab_posts", "ab_outer", "ab_motor", "ab_small") for k in G[g_]]
    ab_hw = H("Hip_HW_post_inserts", "Hip_HW_post_bolts", "Abad_HW_motor", "Abad_HW_small_drum_screws", "Abad_REF_Tensioner", "Abad_HW_hub_inserts")
    # nuts held in printed pockets or slots: pushed in (not turned) before their screws
    captured = lambda k: meta[k]["part"].startswith(("FST_Nut", "Leg4_HW_shin_screws"))
    K = lambda labs, *kinds: [k for k in labs if meta[k]["fit"]["kind"] in kinds]
    shin_unit = list(dict.fromkeys(G["shin"] + P("Leg4_Shin_Plate")))      # (P("Leg4_Shin") already has the plates)
    G["shin"] = shin_unit
    if leg and not abad:
        G["mount"] = G["mount"] + G["blocks"]
    spine_unit = G["spine"] + (G["blocks"] if abad else [])

    # own approach of every hardware item (innermost offset: it is fitted first)
    for k in HW:
        f = meta[k]["fit"]
        d = np.array(f["dir"])
        dist = {"screw": f["len"] + 10.0, "insert": 14.0, "nut": 12.0, "washer": 9.0}.get(f["kind"], 12.0)
        turns = {"screw": min(7.0, max(2.0, f["len"] / 3.0)), "nut": 0.0 if captured(k) else 3.0}.get(
            f["kind"], 2.0 if "tandoff" in k else 0.0)
        F.offset([k], d * dist, turns=turns)
    # sub-assembly steps on the bench (inner offsets of the parts inside a sub-assembly)
    sub = {}
    if abad:
        ox_ = lambda labs: float(np.mean([verts[k][2][0] for k in labs]))
        ref = ox_(G["ab_outer"])
        for g_, dx in (("ab_motor", 70.0), ("ab_small", 70.0), ("ab_hub", 90.0), ("ab_inner", 70.0), ("ab_posts", 60.0)):
            if G[g_]:
                sgn = 1.0 if ox_(G[g_]) >= ref else -1.0
                sub[g_] = (sgn * dx, 0, 0)
                F.offset(G[g_] + pre.get(g_, []), sub[g_])
    if leg:
        F.offset(P("Leg4_Shin_Plate_Back"), y(-40))
        F.offset(P("Leg4_Shin_Plate_Front"), y(40))
    if abad:
        F.offset(G["blocks"], y(40))
    # group offsets (outer): where each part (with its bench hardware) waits
    off = {"motorA": y(140), "motorB": y(140), "moteus": y(-70), "smallA": y(160), "smallB": y(160), "brgA1": y(-60), "brgA2": y(80),
           "drumA": y(220), "cap": y(-60), "drumB": y(240), "posts": (0, 80, 140 if leg else -140), "wall": y(160),
           "bracket": (-140, 0, 0), "bracket_f": (140, 0, 0), "trunnion": (60, 0, 0), "cradle_f": (120, 0, 0),
           "flB": (0, 170, 60 if leg else -60), "armB": y(110), "flA": y(110), "armA": y(120), "shin": y(130), "coupler": y(120), "foot": (0, 0, -80),
           "gposts": y(110), "gantry": y(130), "mount": zz(160) if not abad else y(-160), "stand": y(-400)}
    for g_, v in off.items():
        labs = G[g_] + pre.get(g_, [])
        if g_ == "shin":
            labs = shin_unit + pre["shin"]
        if labs:
            F.offset(labs, v)
    hub_scr = K(H("Abad_HW_hub_bolts"), "screw") if abad else []
    if abad:                                                       # v9.1: the rear bracket joins the ab/ad unit on the bench
        F.offset(ab_unit + ab_hw + G["bracket"] + hub_scr, (-220, 0, 0))
        F.offset(spine_unit + pre["spine"], (420, -140, 0))                  # the bench: off to the side
        F.reach.update({k: 480.0 for k in spine_unit + pre["spine"]})
        F.reach.update({k: 320.0 for k in ab_unit + ab_hw})
    for g_ in pre:
        F.reach.update({k: max(F.reach.get(k, 0.0), 320.0) for k in G[g_] + pre[g_]})
    if leg:
        F.reach.update({k: 320.0 for k in shin_unit})

    F.explode = (1000.0, 1001.0)
    F.hide_at = 3.2
    c0 = np.array([-40.0, 60.0, -40.0])
    az0 = -128 if leg else 52
    F.cam(0.0, c0, 1200 if leg or abad else 900, az0, -14)
    F.cam(3.0, c0, 1150 if leg or abad else 860, az0 + 20, -14)
    F.cards.append((0.2, 3.0, "Capstan v9.1", VARIANT_TITLE[vname]))
    F.t = 3.4
    count = iter(range(1, 200))
    side = -130 if leg else 50
    S = lambda *a, **k: F.step(next(count), *a, **k)

    def view_for(d, az, el=-14.0):
        """Camera azimuth / elevation looking at a fastener from the side it comes in from, leaning
        toward the film's usual side view."""
        a_, e_ = math.radians(az), math.radians(el)
        fs = np.array([math.cos(e_) * math.cos(a_), math.cos(e_) * math.sin(a_), math.sin(e_)])
        dw = R_up @ np.asarray(d, float)
        frm = dw / np.linalg.norm(dw) - 0.8 * fs
        f = -frm / np.linalg.norm(frm)
        return math.degrees(math.atan2(f[1], f[0])), max(-55.0, min(15.0, math.degrees(math.asin(f[2]))))

    def ordered(labs):
        c = np.mean([verts[k][2] for k in labs], axis=0)
        d = np.array(meta[labs[0]]["fit"]["dir"])
        e1 = np.cross(d, [0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.cross(d, [1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(d, e1)
        return sorted(labs, key=lambda k: math.atan2((verts[k][2] - c) @ e2, (verts[k][2] - c) @ e1))

    def fit(head, detail, labs, az=None, context=(), hold=0.9, view=None, see_through=0.15):
        """Fit hardware one item at a time: close on the first, then pull back while the rest follow.
        Within a step: inserts, then washers under the heads, screws, nuts and washers under the nuts."""
        labs = [k for k in labs if k in HWS]
        if not labs:
            return
        az = side if az is None else az
        screws = [k for k in labs if meta[k]["fit"]["kind"] == "screw"]
        sdir = np.array(meta[screws[0]]["fit"]["dir"]) if screws else None
        def rank(k):
            f = meta[k]["fit"]
            if f["kind"] == "washer" and sdir is not None:
                return 1 if np.dot(f["dir"], sdir) > 0 else (0.5 if captured(labs[0]) else 4)
            if f["kind"] == "nut" and captured(k):
                return 0.7
            return {"insert": 0, "other": 1, "screw": 2, "nut": 3}.get(f["kind"], 2)
        groups = {}
        for k in labs:
            groups.setdefault(rank(k), []).append(k)
        seq = [k for r in sorted(groups) for k in ordered(groups[r])]
        n = len(seq)
        stagger = 0.42 if n <= 4 else 0.3 if n <= 8 else 0.2
        start = F.t
        lead = 0.7
        first_mt = 1.7 if meta[seq[0]]["fit"]["kind"] in ("screw", "nut") else 1.2
        t = start + lead
        ends = []
        for i, k in enumerate(seq):
            mt = first_mt if i == 0 else 1.0
            F.move([k], t, t + mt)
            ends.append(t + mt)
            t = t + mt + 0.15 if i == 0 else t + stagger
        end = max(ends)
        for k in context:
            if k not in F.base:
                F.appear[k] = min(F.appear.get(k, 1e9), start)
        f0 = meta[seq[0]]["fit"]
        a_, e_ = view if view is not None else view_for(f0["dir"], az)
        # close on the first item: it and its whole travel fill about half the frame
        travel = np.linalg.norm(F.offsets[seq[0]][0][0])
        p0 = verts[seq[0]][2] + F.offset_at(seq[0], ends[0] + 0.01) + np.array(f0["dir"]) * travel * 0.45
        dist = float(np.clip(2.6 * (f0["len"] + travel) + 60.0, 110.0, 300.0))
        close = (p0, dist, a_ + 4, e_)
        F.cam(start + lead - 0.5, p0, dist, a_, e_)
        F.cam(ends[0] + 0.25, *close)
        # what goes in, from where, and where it ends up: nothing may hide it
        # (fitted, only its outer end: the part it goes into may hide the rest)
        tips = lambda k, t: [verts[k][2] + F.offset_at(k, t) + np.array(meta[k]["fit"]["dir"]) * (meta[k]["fit"]["len"] * 0.5 + 0.3)]
        targets = tips(seq[0], start + lead) + [verts[seq[0]][2] + F.offset_at(seq[0], start + lead)] + tips(seq[0], ends[0] + 0.01)
        F.clear_view(targets, close, ends[0], start, end + hold + 0.3, skip=seq, level=see_through)
        if n > 1:
            wide = F.frame(seq, end, a_ + 8, e_, margin=1.4, extra_t=start + lead)
            wide = (wide[0], max(wide[1], 200.0), wide[2], wide[3])
            F.cam(end + hold, *wide)
            targets = [p for k in seq[1:] for p in tips(k, end + 0.01)]
            F.clear_view(targets, wide, end, ends[0], end + hold + 0.3, skip=seq, level=see_through)
        F.highlight.append((seq, start + lead - 0.4, end + 0.3))
        F.captions.append((start + 0.15, end + hold - 0.1, str(next(count)), head, detail))
        F.t = end + hold

    # --- motors and boards ---
    S("Motors onto the motor plate", "2× EaglePower 8308, boards already soldered on; the wires drop into the open bays",
      [G["motorA"], G["motorB"]], side, -12, margin=1.0, context=plate, stagger=0.6)
    fit("Stator screws", "4× M4 × 12 countersunk per motor, from the plate's back; heads 1 mm below the face", H("FST_Stator"), side + 180)
    fit("moteus inserts", "4× M2.5 heat-set inserts per board, flush in the plate's back", H("FST_Insert_moteus"), side + 180)
    fit("moteus standoffs", "4× M2.5 × 8 male-female standoffs per board, into the inserts", H("FST_Standoff_moteus"), side + 180)
    S("moteus boards", "moteus r4.11 on the standoffs, connectors outboard", [G["moteus"]], side + 180, -10, margin=1.1)
    fit("Board screws", "4× M2.5 × 6 per board", H("FST_Screw_moteus"), side + 180)
    S("Small drums", "Onto each rotor", [G["smallA"], G["smallB"]], side, -15, margin=1.1, context=plate, stagger=0.6)
    fit("Small drum screws", "4× M3 × 10 into each rotor", H("FST_Rotor"))
    # --- drums ---
    S("Drum A bearings", "2× 6806 into the plate's housing, one from each side", [G["brgA1"], G["brgA2"]],
      side - 10, -12, margin=1.4, stagger=0.6)
    fit("Drum A tensioner nuts", "An M4 nylon-insert nut dropped into the slot in each end block, flats down",
        pre["drumA"], side - 20, context=G["drumA"])
    S("Drum A", "Shaft through both bearings; the M5 insert in the shaft end", [G["drumA"] + pre["drumA"]],
      side, -12, margin=1.0, context=plate)
    S("Rear cap", "M5 × 12 into the shaft's insert, from the back; clamps the rear 6806's inner race", [G["cap"]],
      side + 180, -10, margin=2.0)
    fit("Drum B tensioner nuts", "An M4 nylon-insert nut in each end block's slot", pre["drumB"], side - 20, context=G["drumB"])
    S("Drum B", "2× 6806 inside; slides over A's tube", [G["drumB"] + pre["drumB"]], side, -12, margin=1.0, context=L("Big_Drum_A"))
    fit("Drum B journal inserts", "6× M3 heat-set inserts in the journal's end, for flange B", H("FST_Insert_B_hub"))
    st = F.t
    F.rope += [("A", st + 0.5, st + 3.5), ("B", st + 3.8, st + 6.8)]
    F.cam(st + 0.6, np.array([-30.0, 60.0, -40.0]), 520, side + 90, -10)
    F.cam(st + 6.8, np.array([30.0, 75.0, -40.0]), 520, side - 10, -10)
    F.captions.append((st + 0.2, st + 7.0, str(next(count)), "Ropes", "Crossed loops: each end in through a rim hole, across the peg's well and out of the drum's end face"))
    F.t = st + 7.2
    fit("Tensioner pegs", "M4 × 18 in from the rim at each end, the rope through its cross-hole; turned to wind it tight (3 mm key)",
        K(H("REF_Tensioner"), "screw"), side + 40)
    # --- frame ---
    fit("Post and rail inserts", "M4 heat-set inserts in both ends of each post and rail; 2 more on top of each rail for the mount",
        pre["posts"], side, context=G["posts"])
    S("Posts and top rails", "Onto the motor plate", [G["posts"] + pre["posts"]], side, -12, margin=1.0, context=plate)
    S("Front wall", "Tip bearings and the 6808 for drum B pressed in; slides over the drum journal", [G["wall"]], side, -12, margin=1.0)
    if gantry:
        fit("Gantry inserts", "2× M4 heat-set inserts in the wall's front face", H("FST_Insert_gantry"))
    if abad:
        S("Front end bracket", "For the trunnion; its ears sit on the plate and the wall at the right posts (the rear one goes on "
          "with the ab/ad unit)", [G["bracket_f"]], side - 20, -10, margin=1.2)
        fit("Trunnion inserts", "4× M4 heat-set inserts in the front bracket's web", H("Hip_HW_trunnion_inserts"), side - 60)
        S("Trunnion", "The Ø30 journal the cradle's front bearing runs on", [G["trunnion"]], side - 60, -10, margin=1.6)
        fit("Trunnion screws", "4× M4 × 10 countersunk into the inserts", H("Hip_HW_trunnion_screws"), side - 60)
    else:
        S("End bracket", "Slides in from the side; its ears sit on the plate and the wall at the left posts", [G["bracket"]],
          side - 20, -10, margin=1.2)
    ear = "the brackets' ears" if abad else "the bracket's ears"
    left_post = lambda k: verts[k][2][0] < 0.0                      # the rear (ab/ad) bracket's corner posts
    post_scr = H("FST_Screw_posts_back") + H("FST_Bolt_posts")
    later = [k for k in post_scr if left_post(k)] if abad else []
    fit("Motor plate screws", f"M4 from the back into the posts' inserts (M4 × 16; × 25 through {ear})",
        [k for k in H("FST_Screw_posts_back") if k not in later], side + 180)
    fit("Front wall screws", f"M4 through the wall into the posts' other inserts (× 25 through {ear})",
        [k for k in H("FST_Bolt_posts") if k not in later])
    # --- outputs and leg ---
    fit("Flange B inserts", "6× M3 heat-set inserts into the flange's front face", pre["flB"], side + 100, context=G["flB"])
    S("Flange B", "Its hub clamps the 6808's inner race against drum B's shoulder", [G["flB"] + pre["flB"]], side - 10, -10, margin=1.3)
    fit("Flange B screws", "6× M3 × 16 into the journal's inserts", H("FST_Screw_B_flange"))
    S("Thigh" if leg else "Test arm B", "Onto flange B", [G["armB"]], side, -10, margin=1.0)
    fit("Thigh screws" if leg else "Arm B screws", "6× M3 × 10 into the flange's inserts", H("FST_Screw_B_arm"))
    fit("Drum A nose inserts", "4× M3 heat-set inserts in the end of A's nose", H("FST_Insert_A_nose"))
    S("Flange A", "The double-D on A's nose drives it", [G["flA"]], side - 20, -8, margin=1.6)
    fit("Flange A screws", "4× M3 × 10 into the nose's inserts (they only clamp)", H("FST_Screw_A_flange"))
    fit("Flange A inserts", "6× M3 heat-set inserts in the flange's front face", H("FST_Insert_A_flange"))
    S("Crank" if leg else "Test arm A", "Onto flange A" + ("; its stub will carry the gantry bearing" if gantry else ""),
      [G["armA"]], side - 10, -8, margin=1.3)
    fit("Crank screws" if leg else "Arm A screws", "6× M3 × 10 into flange A's inserts", H("FST_Screw_A_arm"))
    if leg:
        S("Shin sandwich", "On the bench: the red core between two black plates, 6000ZZ at the knee, 636ZZ at the lever",
          [P("Leg4_Shin_Plate_Back"), P("Leg4_Shin_Plate_Front")], side + 10, -8, margin=1.3, context=G["shin"],
          frame_context=True, stagger=0.6)
        fit("Shin inserts", "3× short M3 heat-set inserts into the back plate's outer face (the screws pull them deeper)", K(pre["shin"], "insert"), side + 10)
        fit("Shin screws", "3× M3 × 16 countersunk from the front, into the back plate's inserts", K(pre["shin"], "screw"), side + 10)
        S("Shin", "Its knee 6000 lines up with the thigh's Ø10 knee hole", [shin_unit + pre["shin"]], side + 5, -8, margin=1.0)
        fit("Knee pin", "Ø10 shoulder screw (ISO 7379, M8 thread), Ø20 washer, Ø13 spacer on the 6000's inner race; M8 nylon-insert nut at the back",
            H("Leg4_HW_knee_pin"))
        S("Coupler", "Its 636 on the crank pin; the far end on the shin's lever; the rib faces the crank", [G["coupler"]], side, -8, margin=1.1)
        fit("Crank pin", "M6 × 25, washer, Ø9 spacer and shim, nylon-insert nut", H("Leg4_HW_crank_pin"))
        fit("Lever pin", "M6 × 35, Ø9 spacers either side of the lever's 636, nylon-insert nut", H("Leg4_HW_lever_pin"))
        S("Bulb foot", "Slips over the shin's claw", [G["foot"]], side, -8, margin=1.6)
        fit("Foot screws", "2× M3 × 40 through the bulb and the claw's prongs, nylon-insert nuts", H("Leg4_HW_foot_screws"))
    if gantry:
        S("Gantry posts", "Ø20 posts on the wall's inserts", [G["gposts"]], side - 20, -8, margin=1.4)
        S("Front gantry", "In the coupler's plane; its 6902 slides onto the crank's (or arm A's) stub",
          [G["gantry"]], side - 20, -8, margin=1.3)
        fit("Gantry bolts", "2× M4 × 55 through the gantry and its posts into the wall's inserts", H("FST_Bolt_gantry"))
    # --- ab/ad unit (on the bench, then onto the rear bracket) ---
    if abad:
        S("Ab/ad unit, on the bench: motor C", "Onto the ab/ad outer plate, board soldered on; the wires drop into the open bay", [P("Abad_Motor")], side + 90, -10, margin=1.2,
          context=G["ab_outer"], frame_context=True)
        fit("Motor C stator screws", "4× M4 × 12 countersunk, from the plate's outer face", H("Abad_HW_motor_stator"), side + 90)
        fit("Motor C moteus inserts", "4× M2.5 heat-set inserts, flush in the plate's outer face",
            H("Abad_HW_motor_moteus_inserts"), side + 90)
        fit("Motor C moteus standoffs", "4× M2.5 × 8 male-female standoffs into the inserts",
            H("Abad_HW_motor_moteus_standoffs"), side + 90)
        S("moteus C", "On the standoffs, connectors outboard", [P("Abad_moteus")], side + 90, -10, margin=1.3)
        fit("moteus C screws", "4× M2.5 × 6 through the board into the standoffs", H("Abad_HW_motor_moteus_screws"), side + 90)
        S("Small drum C and tip bearing", "", [G["ab_small"]], side + 90, -10, margin=1.4)
        fit("Small drum C screws", "4× M3 × 10 into the rotor", H("Abad_HW_small_drum_screws"), side + 90)
        fit("Ab/ad tensioner nuts", "An M4 nylon-insert nut in each end block's slot", K(pre["ab_hub"], "nut"), side + 90,
            context=G["ab_hub"])
        fit("Hub inserts", "8× M4 heat-set inserts into the hub's end face, on the bench", K(pre["ab_hub"], "insert"), side + 90,
            context=G["ab_hub"])
        S("Ab/ad sector on its two 6808s", "The hollow hub runs in both plates", [G["ab_hub"] + pre["ab_hub"]],
          side + 90, -10, margin=1.1)
        S("Inner plate", "Over the hub's 6808", [G["ab_inner"]], side + 90, -10, margin=1.1)
        fit("Ab/ad post inserts", "2× M4 heat-set inserts in the inner plate", H("Hip_HW_post_inserts"), side + 90)
        S("Ab/ad posts", "Between the two plates", [G["ab_posts"]], side + 90, -10, margin=1.2)
        fit("Ab/ad post bolts", "2× M4 × 70 from the outer plate into the inner plate's inserts", H("Hip_HW_post_bolts"), side + 90)
        fit("Ab/ad tensioner pegs", "M4 × 18 in from the rim at each end, wound tight", K(H("Abad_REF_Tensioner"), "screw"), side + 90)
        S("Rear end bracket onto the hub, on the bench", "Its web against the hub's end; the bracket is still loose, so its inner "
          "face is open", [G["bracket"]], side + 90, -10, margin=1.2)
        fit("Hub screws", "8× M4 × 16 through the bracket's web into the hub's inserts: driven from the open side, before the "
            "module is there", hub_scr, view=(90.0, -22.0))
        S("Ab/ad unit and rear bracket onto the module", "The module's left posts slide into the bracket's ears",
          [ab_unit + ab_hw + G["bracket"] + hub_scr], side + 90, -10, margin=1.1, move_time=1.8)
        fit("Left post screws", "M4 × 25 through the rear bracket's ears into the left posts' inserts, front and back", later,
            side + 180)
        F.abad_rope_t = F.t - 0.5
        S("Cradle front plate", "6806 pressed in; slides over the trunnion, so the module turns on bearings at both ends",
          [G["cradle_f"]], side - 60, -10, margin=1.3, move_time=1.4)
    # --- mount ---
    if abad:
        if leg:
            S("Spine and rail blocks", "On the bench: 2 MGN12H blocks onto the spine's front", [G["blocks"]], side + 30, -10,
              margin=1.2, context=G["spine"], frame_context=True)
            fit("Block screws", "8× M3 × 14 from the spine's front, before the spine goes on", pre["spine"], side + 30)
        S("Spine", "Behind the module, on the three cradle plates' tabs", [spine_unit + (pre["spine"] if leg else [])],
          side + 30, -12, margin=1.0, move_time=1.6)
        if not leg:
            S("Desk angles", "One at each end, behind the spine", [G["mount"]], side + 30, -12, margin=1.0, move_time=1.6)
        fit("Spine screws", ("6× M4 × 16 into the tabs' inserts" if leg else
                             "6× M4 × 25 through the angles and the spine into the tabs' inserts"), H("Hip_HW_rail_screws"), side + 180)
    else:
        if pre["mount"]:
            fit("Block screws", "8× M3 × 12: 2 MGN12H blocks onto the rail carriage", pre["mount"], side + 30, context=G["mount"])
        S({(False, False): "Desk base", (True, False): "Rail carriage"}[(leg, False)], "Onto the two top rails",
          [G["mount"] + pre["mount"]], side + 30, -12, margin=1.0, move_time=1.6)
        fit("Mount screws", "4× M4 into the top rails' inserts", H("Mount_HW_base_screws", "Mount_HW_carriage_screws"), side + 30)
    if not leg:
        fit("Wood screws", "Into the desk, outside the module", H("Mount_HW_wood_screws"), side + 30)
    if leg:
        S("Rail stand", "MGN12 rail on a 2040 upright: the hip slides up and down freely", [G["stand"]], side + 30, -12,
          margin=1.0, move_time=1.8)
    st = F.t
    if not leg:
        F.desk = (st, st + 1.5)
    F.cam(st + 1.0, c0, 1250 if leg else 1000, side + 10, -10)
    run = st + 1.6
    amp = 25.0
    F.motion += [("A", run, run + 5.0, amp, 1.5), ("B", run + 5.5, run + 10.5, amp, 1.5),
                 ("A", run + 11.0, run + 16.0, amp, 1.0), ("B", run + 11.0, run + 16.0, -amp, 1.0)]
    end = run + 16.5
    if abad:
        abfn = lambda t: math.radians(18.0) * window(t, end, end + 6.0, 0.8) * math.sin(2 * math.pi * (t - end) / 3.0)
        F.abad_fn = abfn
        base_angles = Film.angles
        F.angle_fn = lambda t: dict(base_angles(_NoFn(F), t), C=abfn(t))
        end += 6.5
    F.readouts.append((run, end, ["A", "B", "C"] if abad else ["A", "B"]))
    F.captions.append((run + 0.2, run + 10.4, "", "Each output on its own", "8:1 through one crossed rope loop per stage"))
    F.captions.append((run + 11.0, run + 16.0, "", "Both outputs together", "A and B turn about the same axis"))
    if abad:
        F.captions.append((run + 16.5, end, "", "Hip ab/ad", "Motor C turns the whole module about the body axis"))
    F.cam(end, c0, 1250 if leg else 1000, side - 70, -10)
    F.cards.append((end + 0.2, end + 3.5, "Capstan v9.1", "Every part above is in print/, step/ and BOM.md"))
    F.duration = end + 3.7
    return F


class _NoFn:
    """Film view without angle_fn, so Film.angles evaluates the motion list."""
    def __init__(self, f):
        self.angle_fn, self.motion = None, f.motion


def gif_film(meta, verts, view):
    """6 s seamless loop of both outputs working, from one view."""
    if view == "leg_hop":
        return leg_film(meta, verts, part="hops")
    if view == "leg4_hop":
        return leg_film(meta, verts, part="hops", variant="4bar")
    if view == "leg3dof_hop":
        return leg_film(meta, verts, part="hops", variant="3dof")
    F = Film(meta, verts)
    F.theme = "dark"
    L = lambda *n: labels_of(meta, *n)
    F.base = set(meta)
    F.duration = 6.0
    F.motion += [("A", 0.0, 6.0, 26, 2), ("B", 0.0, 6.0, -22, 2)]
    F.rope = [("A", -2.0, -1.0), ("B", -2.0, -1.0)]
    F.beads.append((0.0, 6.0))
    frame = L("Front_Wall", "Post", "FST_Bolt_posts", "FST_Nut_posts", "Base_Mount", "FST_Screw_base",
              "FST_Insert_ribs", "REF_Bearing_15x28_tip", "REF_Bearing_40x52_wall")
    cams = {"overview": (np.array([0, 50, -150]), 680, -128, -12),
            "rear": (np.array([0, 0, -60]), 600, 58, -10),
            "side": (np.array([0, 55, -140]), 680, 0, -5),
            "top": (np.array([0, 60, -75]), 500, -95, -70),
            "capstan_A": (np.array([MA[0], 45, MA[1] - 5]), 330, -160, -18),
            "capstan_B": (np.array([MB[0] - 20, 70, MB[1] - 30]), 380, -20, -24)}
    F.cam(0.0, *cams[view])
    if view.startswith("capstan"):
        F.ghost.append((frame, -1.0, 7.0, 0.12))
        F.readouts.append((0.0, 6.0, [view[-1]]))
    else:
        F.readouts.append((0.0, 6.0, ["A", "B"]))
    return F


GIF_VIEWS = ("leg3dof_hop", "capstan_A", "capstan_B")


def make_gif(mp4, gif, width=720, fps=15):
    pal = gif.with_suffix(".png")
    vf = f"fps={fps},scale={width}:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4), "-vf", vf + ",palettegen=stats_mode=diff", str(pal)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4), "-i", str(pal), "-lavfi",
                    vf + " [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=4", "-loop", "0", str(gif)], check=True)
    pal.unlink()


def rot_xz(theta, c):
    """Rotation by theta in the XZ plane (X toward Z) about an axis along Y through c."""
    q = np.array([math.cos(theta / 2), 0.0, -math.sin(theta / 2), 0.0])
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return c - R.reshape(3, 3) @ c, q


def render(film, model, meta, out_path, title, preview=False, at=None, notext_path=None):
    global UP_R, UP_T
    up0 = (UP_R, UP_T)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, H, W, max_geom=8000)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 1, 1, 0]
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    body_ids = {k: model.body(k).id for k in meta}
    mocap = {k: model.body_mocapid[i] for k, i in body_ids.items()}
    geom_of = {k: model.body_geomadr[i] for k, i in body_ids.items()}
    mat_of = {k: model.geom_matid[g] for k, g in geom_of.items()}
    base_rgba = {k: model.mat_rgba[m].copy() for k, m in mat_of.items()}
    desk_id = model.body_mocapid[model.body("desk").id]
    desk_mat = model.geom_matid[model.body_geomadr[model.body("desk").id]]
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    floor_mat = model.geom_matid[floor_id] if floor_id >= 0 else None
    spin_of = {}
    for group, prefixes in SPIN.items():
        for k in meta:
            if any(k.startswith(p) for p in prefixes):
                spin_of[k] = group
    centers = {"A": np.array([OUT[0], 0, OUT[1]]), "B": np.array([OUT[0], 0, OUT[1]]),
               "mA": np.array([MA[0], 0, MA[1]]), "mB": np.array([MB[0], 0, MB[1]])}

    quat = np.zeros(4)
    n_frames = int(film.duration * FPS)
    frames = [int(x * FPS) for x in at] if at else range(0, n_frames, FPS * 2 if preview else 1)
    proc = None
    if not preview:
        proc = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                                 "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                                 "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                                 "-movflags", "+faststart", str(out_path)], stdin=subprocess.PIPE)
    proc2 = None
    if notext_path is not None and not preview:
        proc2 = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                                  "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                                  "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                                  "-movflags", "+faststart", str(notext_path)], stdin=subprocess.PIPE)
    for fi in frames:
        t = fi / FPS
        if film.up_at is not None:
            UP_R, UP_T = film.up_at(t)
        th = film.angles(t)
        ang = {"A": th["A"], "B": th["B"], "mA": -RATIO * th["A"], "mB": -RATIO * th["B"]}
        phi_ab = film.abad_fn(t) if film.abad_fn is not None else 0.0
        Rab = rot_x(phi_ab)
        for k in meta:
            R, p = film.pose_at(k, t)
            if k in film.custom:
                R, p = film.custom[k](t)
            elif k in ABAD_SPIN:                                  # motor C: turns -RATIO times the ab/ad
                Rm, cm = rot_x(-RATIO * phi_ab), AB_C + np.array([0.0, 0.0, 84.0])
                R, p = Rm @ R, Rm @ (p - cm) + cm
            elif k in spin_of:
                dp, dq = rot_xz(ang[spin_of[k]], centers[spin_of[k]])
                Rd = np.zeros(9)
                mujoco.mju_quat2Mat(Rd, dq)
                Rd = Rd.reshape(3, 3)
                R, p = Rd @ R, Rd @ p + dp
            if film.abad_fn is not None and meta[k].get("abad") != "fixed" and not k.startswith("Stand_"):
                R, p = Rab @ R, Rab @ (p - AB_C) + AB_C               # the module hangs from the ab/ad axis
            R, p = UP_R @ R, UP_R @ p + UP_T
            mujoco.mju_mat2Quat(quat, R.flatten())
            data.mocap_pos[mocap[k]] = p
            data.mocap_quat[mocap[k]] = quat
            a = film.alpha(k, t)
            model.geom_group[geom_of[k]] = 5 if a <= 0.01 else 0
            rgba = base_rgba[k].copy()
            rgba[3] = a
            model.mat_rgba[mat_of[k]] = rgba
            glow = 0.0
            for labs, g0, g1 in film.highlight:
                if k in labs:
                    glow = max(glow, 0.22 * window(t, g0, g1, 0.4))
            model.mat_emission[mat_of[k]] = glow
        if film.desk is not None and t >= film.desk[0]:     # desk slides in under the base from behind
            data.mocap_pos[desk_id] = (0, -500 * (1 - ease((t - film.desk[0]) / (film.desk[1] - film.desk[0]))), 0)
        else:
            data.mocap_pos[desk_id] = (0, 0, -5000)
        if film.desk_ghost:
            model.mat_rgba[desk_mat, 3] = 1.0 - 0.8 * window(t, *film.desk_ghost, 0.6)
        if floor_mat is not None and film.floor_ghost:
            model.mat_rgba[floor_mat, 3] = 1.0 - 0.9 * max(window(t, *w, 0.6) for w in film.floor_ghost)
        mujoco.mj_forward(model, data)              # also places the lights

        lookat, dist, az, el = film.camera(t)
        lookat = np.asarray(lookat, float) if film.world_cam else to_world(lookat)[0]
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, dist, az, el
        renderer.update_scene(data, camera=cam, scene_option=opt)
        scn = renderer.scene
        beads = any(b0 <= t <= b1 for b0, b1 in film.beads)
        for stage, motor, y_mid in (("A", MA, A_MID), ("B", MB, B_MID)):
            frac = 0.0
            f0, f1 = film.rope_fade
            if f0 is not None and t < f1:
                frac = 1.0 if t < f0 else 0.0
            if film.rope_return and t >= film.rope_return[0]:
                frac = ease((t - film.rope_return[0]) / (film.rope_return[1] - film.rope_return[0]))
            for s, r0, r1 in film.rope:
                if s == stage and t >= r0:
                    frac = min(1.0, (t - r0) / (r1 - r0))
            if frac > 0:
                shift = ({s: film.offset_at(SLIDER[(stage, -s)], t) for s in (1, -1)}     # (v5 sliders; v6 pegs: none)
                         if SLIDER[(stage, 1)] in film.offsets else None)
                pts = rope_path(motor, y_mid, th[stage], shift)
                if film.abad_fn is not None:
                    pts = (Rab @ (pts - AB_C).T).T + AB_C
                draw_rope(scn, to_world(pts), frac, 1.3, beads)
        if film.abad_fn is not None and t >= film.abad_rope_t:   # the ab/ad capstan's own rope
            draw_rope(scn, to_world(ab_map(rope_path(MB, AB_MID, -phi_ab, ab=True))), 1.0, 1.3, beads)   # the map mirrors the sense
        if film.extras is not None:
            film.extras(scn, t)
        img = Image.fromarray(renderer.render())
        raw = img
        img = overlay(img, film, t, (np.array(lookat), dist, az, el), title, th)
        if proc2 is not None:
            proc2.stdin.write(raw.convert("RGB").tobytes())
        if preview:
            img.save(out_path.parent / f"{out_path.stem}_{fi:05d}.jpg", quality=85)
        else:
            proc.stdin.write(img.tobytes())
        if fi % (FPS * 10) == 0:
            print(f"  {out_path.name}: {t:5.1f}/{film.duration:.1f} s", flush=True)
    if proc:
        proc.stdin.close()
        proc.wait()
    if proc2 is not None:
        proc2.stdin.close()
        proc2.wait()
    renderer.close()
    UP_R, UP_T = up0


def load(mesh_dir, meta):
    models = {th: build_model(mesh_dir, meta, th) for th in THEMES}
    model = models["light"]
    verts = {}
    for k in meta:
        g = model.body_geomadr[model.body(k).id]
        m = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.geom_quat[g])
        w = v @ R.reshape(3, 3).T + model.geom_pos[g]
        verts[k] = np.array([w.min(0), w.max(0), (w.min(0) + w.max(0)) / 2])
    return models, verts


def load_tris(model, meta):
    """label -> (N, 3, 3) triangles at rest in the model frame (for the films' line-of-sight checks)."""
    out = {}
    for k in meta:
        g = model.body_geomadr[model.body(k).id]
        m = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]]
        fc = model.mesh_face[model.mesh_faceadr[m]:model.mesh_faceadr[m] + model.mesh_facenum[m]]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.geom_quat[g])
        out[k] = (v @ R.reshape(3, 3).T + model.geom_pos[g])[fc]
    return out


def main_variants(mesh_dir, jobs, preview, at):
    """jobs: build_<variant> and/or jump_<variant> (leg builds). Writes video/<job>.mp4 and <job>_notext.mp4."""
    out = VIDEO if not preview else mesh_dir.parent / "preview"
    out.mkdir(exist_ok=True)
    for job in jobs:
        kind, vname = job.split("_", 1)
        meta = json.loads((mesh_dir / f"{vname}.json").read_text())
        models, verts = load(mesh_dir, meta)
        if kind == "build":
            film = author_variant(meta, verts, vname, load_tris(models["light"], meta))
            title = f"Capstan v9.1  ·  {VARIANT_TITLE[vname]}  ·  build"
        else:
            film = leg_film(meta, verts, variant="3dof" if "abad" in vname else "4bar")
            title = f"Capstan v9.1  ·  {VARIANT_TITLE[vname]}  ·  jumping"
        print(f"{job}: {film.duration:.1f} s", flush=True)
        render(film, models[film.theme], meta, out / f"{job}.mp4", title, preview, at, notext_path=out / f"{job}_notext.mp4")


def main():
    mesh_dir = Path(sys.argv[1]).resolve()
    at = [float(x) for a in sys.argv if a.startswith("--at=") for x in a[5:].split(",")]
    preview = "--preview" in sys.argv or bool(at)
    only = [a for a in sys.argv[2:] if not a.startswith("--")]
    if "--variants" in sys.argv:
        return main_variants(mesh_dir, [a for a in sys.argv[2:] if not a.startswith("--")], preview, at)
    meta = json.loads((mesh_dir / "parts.json").read_text())
    models = {th: build_model(mesh_dir, meta, th) for th in THEMES}
    model = models["light"]
    verts = {}
    for k in meta:
        g = model.body_geomadr[model.body(k).id]
        m = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.geom_quat[g])
        w = v @ R.reshape(3, 3).T + model.geom_pos[g]        # mesh frame -> world (body frame = world at rest)
        verts[k] = np.array([w.min(0), w.max(0), (w.min(0) + w.max(0)) / 2])
    out = VIDEO if not preview else mesh_dir.parent / "preview"
    out.mkdir(exist_ok=True)
    jobs = {"v4_build": (author_v4_build, "Stanley-style leg v4  ·  build"),
            "v4_jump": (lambda m, v: leg_film(m, v, variant="3dof"), "Stanley-style leg v4  ·  3 DOF jumping"),
            "assembly": (author_assembly, "Coaxial Capstan v3  ·  assembly"),
            "detailed": (author_detailed, "Coaxial Capstan v3  ·  detailed assembly"),
            "exploded": (author_exploded, "Coaxial Capstan v3  ·  exploded view"),
            "leg": (leg_film, "Coaxial Capstan v3  ·  five-bar leg"),
            "leg_4bar": (lambda m, v: leg_film(m, v, variant="4bar"), "Coaxial Capstan v3  ·  4-bar leg"),
            "leg_3dof": (lambda m, v: leg_film(m, v, variant="3dof"), "Coaxial Capstan v3  ·  3-DOF leg")}
    for name, (author, title) in jobs.items():
        if only and name not in only:
            continue
        film = author(meta, verts)
        print(f"{name}: {film.duration:.1f} s", flush=True)
        for t0, t1, step_no, head, _ in film.captions:
            print(f"   {t0:6.1f}-{t1:6.1f}  {step_no:>2} {head}", flush=True)
        fname = "assembly_detailed" if name == "detailed" else name
        render(film, models[film.theme], meta, out / f"{fname}.mp4", title, preview, at)
    if (not only or "gifs" in only) and not preview:
        gdir = out / "gifs"
        gdir.mkdir(exist_ok=True)
        for view in GIF_VIEWS:
            film = gif_film(meta, verts, view)
            tmp = gdir / f"_{view}.mp4"
            print(f"gif {view}: {film.duration:.1f} s", flush=True)
            render(film, models[film.theme], meta, tmp, f"Coaxial Capstan v3  ·  {view.replace('_', ' ')}")
            make_gif(tmp, gdir / f"{view}.gif")
            tmp.unlink()


if __name__ == "__main__":
    main()
