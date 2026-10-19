"""Quadruped moves film (kinematics only, not a dynamics simulation), on the built robot:

1. Ab/ad: the body sways and rolls over planted feet; side-steps; turns on the spot; balances on each
   diagonal pair with the other two legs lifted and spread (ab/ad out).
2. Terrain: a trot over flat ground, up and down a flight of stairs, then across rough ground (random
   blocks). Footholds are planned on the terrain (kept off the edges), the body follows the ground's
   mean height and pitches with it.

Every foot is placed in 3D: its ab/ad angle comes from where it is across the body, the thigh and knee
from the 4-bar's IK in the tilted leg plane.

Run:  python make_quad_moves.py <quad_mesh_dir> [--at=t1,t2]
Writes ../video/quadruped_moves.mp4 and quadruped_moves_notext.mp4 (or stills with --at).
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import make_quad_film as qf

mv, gq = qf.mv, qf.gq
PODS, FOOT_R, HIP_Z = qf.PODS, qf.FOOT_R, qf.HIP_Z
ROOT = qf.ROOT
DIAG = {"LF": 0.0, "RR": 0.0, "RF": 0.5, "LR": 0.5}          # trot pairs
PERIOD = 0.72
FOOT_LOCAL_Y = gq.g.L4_Y["Leg4_Shin"] + 4.0                  # the leg plane (pod-local Y of the foot)
BOX_H = 400.0                                               # the box jump's platform (Spot steps up ~300)
EDGE_KEEP = 48.0                                            # footholds kept this far off terrain edges


# ------------------------------------------------------------------------------------------------
# Terrain: boxes (x0, x1, y0, y1, top)
# ------------------------------------------------------------------------------------------------
def terrain_boxes():
    boxes = []
    rise, tread, n, landing = 45.0, 250.0, 4, 520.0
    x = -1150.0                                             # stairs up (the robot walks toward -X)
    for k in range(n):
        boxes.append((x - tread * (k + 1), x - tread * k, -420.0, 420.0, rise * (k + 1)))
    top0 = x - tread * n
    boxes.append((top0 - landing, top0, -420.0, 420.0, rise * n))
    top_end = top0 - landing
    for k in range(n - 1):                                  # and down
        boxes.append((top_end - tread * (k + 1), top_end - tread * k, -420.0, 420.0, rise * (n - 1 - k)))
    stairs_end = top_end - tread * (n - 1)
    rng = np.random.default_rng(7)                          # rough ground: 130 mm cells, 0..40 mm
    cell = 130.0
    rx0, rx1 = stairs_end - 450.0, stairs_end - 450.0 - 8 * cell
    for i in range(8):
        for j in range(-3, 3):
            h = float(rng.choice([8.0, 16.0, 24.0, 32.0, 40.0]))
            boxes.append((rx0 - (i + 1) * cell, rx0 - i * cell, j * cell, (j + 1) * cell, h))
    return boxes, dict(stairs=(-1150.0, stairs_end), rough=(rx1, rx0))


BOXES, ZONES = terrain_boxes()


def ground(x, y):
    h = 0.0
    for x0, x1, y0, y1, top in BOXES:
        if x0 <= x <= x1 and y0 <= y <= y1:
            h = max(h, top)
    return h


def ground_smooth(x, y, w=140.0):
    return float(np.mean([ground(x + d, y + e) for d in np.linspace(-w, w, 9) for e in (-w / 2, 0.0, w / 2)]))


def keep_off_edges(x, y):
    xs = sorted({b[0] for b in BOXES} | {b[1] for b in BOXES})
    ys = sorted({b[2] for b in BOXES} | {b[3] for b in BOXES})
    for _ in range(2):
        for e in xs:
            if abs(x - e) < EDGE_KEEP:
                x = e + EDGE_KEEP * (1.0 if x >= e else -1.0)
        for e in ys:
            if abs(y - e) < EDGE_KEEP and ground(x, y) != ground(x, e + (EDGE_KEEP if y < e else -EDGE_KEEP)):
                y = e + EDGE_KEEP * (1.0 if y >= e else -1.0)
    return x, y


# ------------------------------------------------------------------------------------------------
# Body pose
# ------------------------------------------------------------------------------------------------
def rot(yaw=0.0, pitch=0.0, roll=0.0):
    cy, sy, cp, sp, cr, sr = (math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll))
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def sm(u):
    u = max(0.0, min(1.0, u))
    return u * u * (3 - 2 * u)


class Plan:
    """Timeline: body path (x, y, yaw) + extra (dz, roll, pitch), gait segments, planned feet."""

    def __init__(self, meta):
        self.core = {}
        for p in PODS:
            k = f"{p}__Motor_Plate"
            self.core[p] = (np.diag(meta[k]["mirror"]), np.array(meta[k]["shift"]))
        self.hip = {p: self.core[p][0] @ np.array([mv.OUT[0], 0.0, mv.OUT[1]]) + self.core[p][1] for p in PODS}
        self.nom = {p: np.array([self.hip[p][0], (self.core[p][0] @ np.array([0.0, FOOT_LOCAL_Y, 0.0]) + self.core[p][1])[1]])
                    for p in PODS}
        self.h_walk = FOOT_R + float(mv.L4_R0) - HIP_Z
        self.half = abs(self.hip["LF"][0])
        self.chapters, self.captions = [], []
        self.dt = 1.0 / 120.0
        self._build()

    # -- the script --------------------------------------------------------------------------------
    def _build(self):
        segs = []                                           # (T, mode, fn(u) -> dict of path deltas), mode: stand/trot/pose
        t = 0.0

        def add(T, mode, head, detail, **kw):
            nonlocal t
            segs.append(dict(t0=t, t1=t + T, mode=mode, **kw))
            self.captions.append((t + 0.15, t + T - 0.15, "", head, detail))
            t += T

        add(1.6, "stand", "Ready", "Standing on the four feet: ab/ad at 0")
        add(6.0, "stand", "Sway and roll", "Feet planted: the four ab/ad drives roll the body ±10° and shift it ±40 mm",
            sway=True)
        add(PERIOD * 9, "trot", "Side-step", "Ab/ad swings each foot across: the body crabs 220 mm left and back", side=220.0)
        add(PERIOD * 10, "trot", "Turn on the spot", "Ab/ad and hips together: 50° and back", turn=math.radians(50.0))
        add(5.5, "pose", "Two legs: front-left + rear-right", "The other diagonal lifted and spread with ab/ad", lift=("RF", "LR"))
        add(5.5, "pose", "Two legs: front-right + rear-left", "And the other diagonal", lift=("LF", "RR"))
        self.t_terrain = t
        add(0.8, "stand", "Terrain", "Flat ground, a flight of stairs (45 mm rises), rough ground (0–40 mm blocks)")
        self.t_walk0 = t
        # walk to the end of the rough patch at a speed set by the ground under the body
        x_end = ZONES["rough"][0] - 600.0
        x, tt = 0.0, 0.0
        while x > x_end:
            v = self.speed_at(x)
            x -= v * self.dt
            tt += self.dt
        cycles = math.ceil(tt / PERIOD) + 1
        add(PERIOD * cycles, "trot", "Flat, stairs, rough ground", "Footholds planned on the terrain, kept off the edges",
            walk=True)
        add(1.2, "stand", "", "")
        jl = self._jump_params(v=3.2, angle=45.0, slow=0.35)
        add(jl["T"], "jump", "Standing long jump (0.35× speed)",
            f"3.2 m/s at 45°: {jl['flight'] / 1000:.2f} m in the air, {abs(jl['dx']) / 1000:.2f} m of travel (estimate for the ±90° capstans: 3.4 m/s)",
            jump=jl)
        add(1.2, "stand", "", "")
        jb = self._jump_params(v=3.3, angle=65.0, slow=0.35, rise=BOX_H)
        add(jb["T"], "jump", f"Box jump onto {BOX_H / 10:.0f} cm (0.35× speed)",
            "3.3 m/s at 65°, legs tucked over the edge: Spot steps up about 30 cm", jump=jb)
        add(1.8, "stand", "", "")
        self.segs, self.t_end = segs, t
        wi = next(i for i, x in enumerate(segs) if x.get("walk"))
        self.walk_x = self._integrate_walk(segs[wi])
        # the box: its near edge halfway between the front feet at take-off and the hind feet landed
        x0 = float(self.walk_x[-1]) + jl["dx"]
        x1 = x0 + jb["dx"]
        xe = (x0 - self.half + x1 + self.half) / 2
        assert (x0 - self.half) - xe > 40.0, "box jump too short for the body"
        BOXES.append((xe - 1100.0, xe, -450.0, 450.0, BOX_H))
        # captions for the walk by zone (in place of the walk's own and the stand after it)
        cap = self.captions
        del cap[wi:wi + 2]
        w = segs[wi]
        walk_caps = []
        zx = [(0.0, "Flat ground", "Trot, 0.72 s period"), (ZONES["stairs"][0] + 400.0, "Stairs", "Up four 45 mm steps, across the landing, down"),
              (ZONES["rough"][1] + 300.0, "Rough ground", "Random 130 mm blocks, 8 to 40 mm high")]
        ts = []
        for x0, h, d in zx:
            k = int(np.argmax(self.walk_x <= x0 - 1e-6)) if np.any(self.walk_x <= x0 - 1e-6) else 0
            ts.append((w["t0"] + k * self.dt, h, d))
        for (ta, h, d), tb in zip(ts, [x[0] for x in ts[1:]] + [w["t1"]]):
            walk_caps.append((ta + 0.1, tb - 0.1, "", h, d))
        cap[wi:wi] = walk_caps
        self.captions = [c for c in cap if c[3]]
        self._plan_feet()

    def speed_at(self, x):
        s0, s1 = ZONES["stairs"]
        r0, r1 = ZONES["rough"]
        slow = (s1 - 300.0 < x < s0 + 400.0) or (r0 - 300.0 < x < r1 + 400.0)
        return 210.0 if slow else 330.0

    def _integrate_walk(self, seg):
        n = int(round((seg["t1"] - seg["t0"]) / self.dt)) + 2
        xs = np.zeros(n)
        x = 0.0
        x_end = ZONES["rough"][0] - 600.0
        for i in range(n):
            u = i * self.dt
            env = min(1.0, u / PERIOD, max(0.0, (seg["t1"] - seg["t0"] - u)) / PERIOD)
            if x > x_end:
                x -= self.speed_at(x) * env * self.dt
            xs[i] = x
        return xs

    def seg_at(self, t):
        for s in self.segs:
            if s["t0"] <= t < s["t1"]:
                return s
        return self.segs[-1]

    def path(self, t):
        """(x, y, yaw) of the body's planned path (no sway / pitch)."""
        x = y = yaw = 0.0
        for s in self.segs:
            if t < s["t0"]:
                break
            u = min(t, s["t1"]) - s["t0"]
            T = s["t1"] - s["t0"]
            if "side" in s:
                y += s["side"] * (sm(u / (T / 2)) - sm((u - T / 2) / (T / 2))) if u < T else 0.0
            if "turn" in s:
                yaw += s["turn"] * (sm(u / (T / 2)) - sm((u - T / 2) / (T / 2))) if u < T else 0.0
            if "walk" in s:
                x = float(self.walk_x[min(len(self.walk_x) - 1, int(u / self.dt))])
        return x + self.jump_dx(t), y, yaw

    # -- jumps ---------------------------------------------------------------------------------------
    def _jump_params(self, v, angle, slow, flip=0.0, drift=0.0, stroke=125.0, drop=95.0, rise=0.0):
        """Crouch, push (constant acceleration over `stroke` along the launch line), ballistic flight
        (a turn of 2 pi in pitch for a flip), absorb (the push reversed), stand. Push to absorb are shown
        at `slow`; v in m/s, lengths in mm."""
        g = 9810.0
        a = math.radians(angle)
        d = np.array([-math.cos(a), 0.0, math.sin(a)])         # toward -X (the way it walks) and up
        v = v * 1000.0
        vx = d[0] * v + drift * (1 if flip else 0)
        Tp = 2 * stroke / v
        vz = d[2] * v
        Tf = (vz + math.sqrt(vz * vz - 2 * g * rise)) / g        # lands `rise` higher (a box) or level
        J = dict(v=v, d=d, vx=vx, stroke=stroke, drop=drop, flip=flip, slow=slow, Tc=0.9, Tp=Tp, Tf=Tf, Ta=Tp, Tr=1.1,
                 rise=rise, flight=abs(vx * Tf))
        dl = np.array([d[0], 0.0, -d[2]])
        J["dl"] = dl
        J["dx"] = d[0] * stroke + vx * Tf + dl[0] * stroke
        J["T"] = J["Tc"] + (Tp + Tf + Tp) / slow + J["Tr"]
        return J

    def jump_dx(self, t):
        return sum(s["jump"]["dx"] for s in getattr(self, "segs", []) if "jump" in s and t >= s["t1"])

    def jump_dz(self, t):
        return sum(s["jump"]["rise"] for s in getattr(self, "segs", []) if "jump" in s and t >= s["t1"])

    def jump_state(self, s, u):
        """(R, pb, feet) during a jump segment, u = time into it (video)."""
        J = s["jump"]
        x, y, yaw = self.path(s["t0"])
        base = np.array([x, y, self.h_walk + ground_smooth(x, y)])
        feet0 = self.base_feet(s["t0"])
        Pc = base - np.array([0.0, 0.0, J["drop"]])
        Pto = Pc + J["d"] * J["stroke"]
        vel = np.array([J["vx"], 0.0, J["d"][2] * J["v"]])
        Ptd = Pto + np.array([J["vx"] * J["Tf"], 0.0, J["rise"]])
        Plc = Ptd + J["dl"] * J["stroke"]
        Pend = Plc + np.array([0.0, 0.0, J["drop"]])
        land = {p: feet0[p] + np.array([J["dx"], 0.0, J["rise"]]) for p in PODS}
        th0, fl = J["flip"], J["flip"] > 0
        sl = J["slow"]
        t1 = J["Tc"]
        t2 = t1 + J["Tp"] / sl
        t3 = t2 + J["Tf"] / sl
        t4 = t3 + J["Ta"] / sl

        def pitch_at(tf):                                      # through the flight
            return th0 + (2 * math.pi - 2 * th0) * tf / J["Tf"] if fl else 0.0

        R_to, R_td = rot(pitch=th0), rot(pitch=2 * math.pi - th0 if fl else 0.0)
        if u < t1:                                             # crouch
            e = sm(u / t1)
            return rot(), base + (Pc - base) * e, feet0
        if u < t2:                                             # push
            tau = (u - t1) / (t2 - t1)
            return rot(pitch=th0 * tau * tau), Pc + J["d"] * J["stroke"] * tau * tau, feet0
        if u < t3:                                             # flight
            tf = (u - t2) * sl
            P = Pto + vel * tf - np.array([0.0, 0.0, 0.5 * 9810.0 * tf * tf])
            R = rot(pitch=pitch_at(tf))
            w = tf / J["Tf"]
            out = {}
            for p in PODS:
                b0 = R_to.T @ (feet0[p] - Pto)
                b1 = R_td.T @ (land[p] - Ptd)
                fb = b0 + (b1 - b0) * sm(w)
                hip = np.array([self.hip[p][0], fb[1], self.hip[p][2]])
                fb = fb + (hip - fb) * (0.32 if fl else 0.22) * math.sin(math.pi * w)   # tucked mid-air
                out[p] = R @ fb + P
            return R, P, out
        if u < t4:                                             # absorb
            tau = (u - t3) / (t4 - t3)
            e = 2 * tau - tau * tau
            R = rot(pitch=(2 * math.pi - th0 + th0 * e) if fl else 0.0)
            return R, Ptd + J["dl"] * J["stroke"] * e, land
        e = sm((u - t4) / J["Tr"])                             # stand
        return rot(), Plc + (Pend - Plc) * e, land

    def body(self, t):
        """(R, p) of the body in the world."""
        s = self.seg_at(t)
        if "jump" in s:
            R, p, _ = self.jump_state(s, t - s["t0"])
            return R, p
        x, y, yaw = self.path(t)
        u, T = t - s["t0"], s["t1"] - s["t0"]
        dz = roll = pitch = dy = 0.0
        if s.get("sway"):
            env = sm(u / 0.8) * sm((T - u) / 0.8)
            ph = 2 * math.pi * u / 3.0
            dy, roll, dz = 40.0 * math.sin(ph) * env, math.radians(10.0) * math.sin(ph + math.pi / 2) * env, -18.0 * math.sin(2 * ph) * env
        if "lift" in s:
            env = sm(u / 0.9) * sm((T - u) / 0.9)
            # lean a little toward the standing diagonal's line and lower: the lifted pair's side up
            sgn = 1.0 if s["lift"][0] == "RF" else -1.0
            roll, pitch, dz = math.radians(4.0) * sgn * env, math.radians(3.0) * sgn * env, -20.0 * env
        R = rot(yaw, pitch, roll)
        Ryaw = rot(yaw)
        if self.jump_dz(t) > 0.0:                                 # up on the box: follow the planted feet
            ft = self.base_feet(t)
            hz = {p: ft[p][2] - FOOT_R for p in PODS}
        else:
            hz = {p: ground_smooth(*(np.array([x, y]) + Ryaw[:2, :2] @ self.nom[p])) for p in PODS}
        gz = np.mean(list(hz.values()))
        front = np.mean([hz[p] for p in ("LF", "RF")])
        rear = np.mean([hz[p] for p in ("LR", "RR")])
        pitch_g = -math.atan2(front - rear, 2 * self.half)
        R = rot(yaw, pitch + pitch_g, roll)
        p = np.array([x, y, 0.0]) + Ryaw @ np.array([0.0, dy, 0.0]) + np.array([0.0, 0.0, self.h_walk + gz + dz])
        return R, p

    def nominal(self, p, t):
        x, y, yaw = self.path(t)
        q = np.array([x, y]) + rot(yaw)[:2, :2] @ self.nom[p]
        return q

    def foothold(self, p, t):
        x, y = keep_off_edges(*self.nominal(p, t))
        return np.array([x, y, ground(x, y) + FOOT_R])

    # -- feet ----------------------------------------------------------------------------------------
    def _plan_feet(self):
        """Per leg: list of (t0, t1, p0, p1, apex) swings; the foot is planted between them."""
        self.swings = {p: [] for p in PODS}
        self.start = {p: self.foothold(p, 0.0) for p in PODS}
        last = dict(self.start)
        for s in self.segs:
            if s["mode"] != "trot":
                continue
            T = s["t1"] - s["t0"]
            for p in PODS:
                k = 0
                while True:
                    a = s["t0"] + (k + 0.5 - DIAG[p]) * PERIOD if DIAG[p] == 0.0 else s["t0"] + k * PERIOD
                    b = a + PERIOD / 2
                    if b > s["t1"] + 1e-6:
                        break
                    if a >= s["t0"] - 1e-6:
                        p1 = self.foothold(p, min(b + PERIOD / 4, s["t1"]))
                        p0 = last[p]
                        n = 12
                        hmax = max(ground(*(p0[:2] + (p1[:2] - p0[:2]) * i / n)) for i in range(n + 1))
                        apex = max(p0[2], p1[2], hmax + FOOT_R) + 40.0
                        if np.linalg.norm(p1[:2] - p0[:2]) > 6.0 or abs(p1[2] - p0[2]) > 1.0:
                            self.swings[p].append((a, b, p0, p1, apex))
                            last[p] = p1
                    k += 1
        self.final = last

    def base_feet(self, t):
        """Planted / swinging feet from the gait plan, moved on by the jumps already made."""
        out = {}
        dx = np.array([self.jump_dx(t), 0.0, self.jump_dz(t)])
        for p in PODS:
            f = self.start[p]
            for a, b, p0, p1, apex in self.swings[p]:
                if t >= b:
                    f = p1
                elif a <= t < b:
                    w = (t - a) / (b - a)
                    e = sm(w)
                    xy = p0[:2] + (p1[:2] - p0[:2]) * sm(min(1.0, max(0.0, (w - 0.12) / 0.76)))
                    zl = p0[2] + (p1[2] - p0[2]) * e
                    f = np.array([xy[0], xy[1], zl + (apex - (p0[2] + p1[2]) / 2) * math.sin(math.pi * w)])
                    break
                else:
                    break
            out[p] = f + dx
        return out

    def feet(self, t):
        s = self.seg_at(t)
        if "jump" in s:
            return self.jump_state(s, t - s["t0"])[2]
        out = self.base_feet(t)
        if "lift" in s:                                     # the lifted pair: up, out (ab/ad) and a wave of the knee
            u, T = t - s["t0"], s["t1"] - s["t0"]
            env = sm(u / 1.1) * sm((T - u) / 1.1)
            R, pb = self.body(t)
            for p in s["lift"]:
                side = 1.0 if p[0] == "L" else -1.0
                fwd = 1.0 if p[1] == "F" else -1.0
                loc = np.array([self.nom[p][0] + fwd * 40.0, self.nom[p][1] + side * 55.0, HIP_Z - float(mv.L4_R0) + 150.0])
                loc[0] += 35.0 * math.sin(2 * math.pi * u / 1.4) * env
                up = R @ loc + pb
                out[p] = out[p] + (up - out[p]) * env
        return out

    # -- IK ------------------------------------------------------------------------------------------
    def angles(self, p, foot, R, pb):
        """(thigh, shin deg, ab/ad rad) of pod p for a world foot point."""
        D, sh = self.core[p]
        loc = D @ (R.T @ (np.asarray(foot) - pb) - sh)
        dy, dz = loc[1] - mv.AB_C[1], loc[2] - mv.AB_C[2]
        l = FOOT_LOCAL_Y - mv.AB_C[1]
        r = math.hypot(dy, dz)
        zz = -math.sqrt(max(r * r - l * l, 1.0))
        ab = math.atan2(dz, dy) - math.atan2(zz, l)
        ab = (ab + math.pi) % (2 * math.pi) - math.pi
        x, z = loc[0], zz + mv.AB_C[2]
        ddx, ddz = x - mv.OUT[0], z - mv.OUT[1]
        rr = min(math.hypot(ddx, ddz), mv.L4_L1 + mv.L4_L2 - 2.0)
        qt, qs = mv.leg4_ik(rr, math.degrees(math.atan2(ddz, ddx)))
        return qt, qs, ab, math.hypot(ddx, ddz)


def main():
    qdir = Path(sys.argv[1]).resolve()
    at = [float(x) for a in sys.argv if a.startswith("--at=") for x in a[5:].split(",")]
    meta = json.loads((qdir / "quad.json").read_text())["meta"]
    P = Plan(meta)
    # the scene: make_quad_film's, with the terrain added to the world
    boxes = "".join(f'<geom type="box" size="{(x1 - x0) / 2} {(y1 - y0) / 2} {top / 2}" pos="{(x0 + x1) / 2} {(y0 + y1) / 2} {top / 2 - 0.5}" '
                    f'material="terrain{int(top) % 3}" contype="0" conaffinity="0"/>' for x0, x1, y0, y1, top in BOXES)
    mats = ('<material name="terrain0" rgba="0.42 0.40 0.37 1" specular="0.1" reflectance="0.05"/>'
            '<material name="terrain1" rgba="0.47 0.45 0.41 1" specular="0.1" reflectance="0.05"/>'
            '<material name="terrain2" rgba="0.37 0.35 0.33 1" specular="0.1" reflectance="0.05"/>')
    orig = qf.mujoco.MjModel.from_xml_string
    captured = {}

    def patched(xml):
        xml = xml.replace('size="6000 6000 1"', 'size="20000 20000 1"').replace('texrepeat="40 40"', 'texrepeat="133 133"')
        xml = xml.replace("<asset>", "<asset>" + mats, 1).replace("</worldbody>", boxes + "</worldbody>", 1)
        captured["xml"] = xml
        return orig(xml)

    qf.mujoco.MjModel.from_xml_string = patched
    model = qf.build_model(meta)
    qf.mujoco.MjModel.from_xml_string = orig
    verts, _ = qf.mesh_data(model, meta)

    # check the plan: ab/ad and leg reach over the whole film
    worst = {"ab": 0.0, "rmax": 0.0, "rmin": 1e9, "out": 0.0, "kmin": 0.0, "kmax": 0.0}
    for t in np.arange(0.0, P.t_end, 1 / 30):
        R, pb = P.body(t)
        ft = P.feet(t)
        for p in PODS:
            qt, qs, ab, r = P.angles(p, ft[p], R, pb)
            worst["ab"] = max(worst["ab"], abs(math.degrees(ab)))
            worst["rmax"], worst["rmin"] = max(worst["rmax"], r), min(worst["rmin"], r)
            ta, tb = qs - mv.L4_Q0["shin"], qt - mv.L4_Q0["thigh"]
            worst["out"] = max(worst["out"], abs(ta), abs(tb))
            worst["kmin"], worst["kmax"] = min(worst["kmin"], tb - ta), max(worst["kmax"], tb - ta)
    print(f"plan: {P.t_end:.1f} s; max |ab/ad| {worst['ab']:.1f} deg (limit {gq.AB_LIMIT}); hip-foot {worst['rmin']:.0f}..{worst['rmax']:.0f} mm "
          f"(reach {mv.L4_L1 + mv.L4_L2:.0f}); outputs up to {worst['out']:.1f} deg (travel {gq.g.OUT_RANGE / 2:.0f}), "
          f"knee {worst['kmin']:+.0f}..{worst['kmax']:+.0f} deg (range -60..+60)", flush=True)

    F = mv.Film(meta, verts)
    F.captions = list(P.captions)
    F.cards.append((P.t_end + 0.2, P.t_end + 2.9, "Capstan quadruped", "CAD animation  ·  kinematics only, not a dynamics simulation"))
    dur = P.t_end + 3.0
    F.duration = dur

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, mv.H, mv.W, max_geom=20000)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 1, 1, 0]
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    bid = {k: model.body(k).id for k in meta}
    mocap = {k: model.body_mocapid[i] for k, i in bid.items()}
    geom_of = {k: model.body_geomadr[i] for k, i in bid.items()}
    pods = {k: (qf.classify(meta[k]), np.diag(meta[k]["mirror"]), np.array(meta[k]["shift"])) for k in meta if "pod" in meta[k]}
    for k in meta:
        if k == "Q_Build_Stand":
            model.geom_group[geom_of[k]] = 5
    # camera keys: (t, az, el, dist, lookat offset)
    T_ter = P.t_terrain
    keys = [(0.0, -60.0, -16.0, 1700.0), (7.6, -75.0, -10.0, 1600.0), (14.0, -40.0, -22.0, 1750.0), (21.0, -120.0, -18.0, 1700.0),
            (T_ter - 0.5, -150.0, -14.0, 1700.0), (T_ter + 1.0, -100.0, -12.0, 1900.0)]
    jumps = [x for x in P.segs if "jump" in x]
    keys += [(jumps[0]["t0"] - 2.0, -60.0, -14.0, 1900.0), (jumps[0]["t0"] - 0.3, -90.0, -6.0, 2300.0),
             (jumps[-1]["t1"], -90.0, -6.0, 2300.0), (P.t_end, -70.0, -12.0, 2100.0), (dur, -55.0, -16.0, 2100.0)]

    def camera(t, pb):
        k = keys[-1]
        for a, b in zip(keys, keys[1:]):
            if a[0] <= t <= b[0]:
                u = mv.ease((t - a[0]) / (b[0] - a[0]))
                k = tuple(x + (y - x) * u for x, y in zip(a, b))
                break
        x, y, _ = P.path(t)
        x = pb[0]
        base = P.h_walk + ground_smooth(x, y)
        return np.array([x, y * 0.6, base - 140.0 + 0.75 * max(0.0, pb[2] - base)]), k[3], k[1], k[2]

    quat = np.zeros(4)
    frames = [int(x * mv.FPS) for x in at] if at else range(int(dur * mv.FPS))
    out = ROOT / "video" / "quadruped_moves.mp4"
    if at:
        out = qdir.parent / "preview" / "quadruped_moves.mp4"
        out.parent.mkdir(exist_ok=True)
    procs = []
    if not at:
        for path in (out, out.with_name(out.stem + "_notext.mp4")):
            procs.append(subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                           "-s", f"{mv.W}x{mv.H}", "-r", str(mv.FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                                           "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE))
    title = "Capstan quadruped  ·  ab/ad moves and terrain"
    for fi in frames:
        t = min(fi / mv.FPS, P.t_end)
        R, pb = P.body(t)
        feet = P.feet(t)
        states = {}
        for p in PODS:
            qt, qs, ab, _ = P.angles(p, feet[p], R, pb)
            states[p] = qf.leg_state(qt, qs, ab)
        for k in meta:
            if k in pods:
                (kind, rides), D, sh = pods[k]
                Rk, pk = qf.pod_pose(kind, rides, D, sh, states[meta[k]["pod"]])
            else:
                Rk, pk = np.eye(3), np.zeros(3)
            data.mocap_pos[mocap[k]] = R @ pk + pb
            mujoco.mju_mat2Quat(quat, (R @ Rk).flatten())
            data.mocap_quat[mocap[k]] = quat
        mujoco.mj_forward(model, data)
        lookat, dist, az, el = camera(fi / mv.FPS, pb)
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, dist, az, el
        renderer.update_scene(data, camera=cam, scene_option=opt)
        scn = renderer.scene
        for p in PODS:                                       # capstan ropes
            st = states[p]
            kc, kd = f"{p}__Motor_Plate", f"{p}__Abad_Drum"
            D, sh = pods[kc][1], pods[kc][2]
            for stage, motor, y_mid in (("A", mv.MA, mv.A_MID), ("B", mv.MB, mv.B_MID)):
                pts = mv.rope_path(motor, y_mid, st["th"][stage])
                pts = (st["Rab"] @ (pts - mv.AB_C).T).T + mv.AB_C
                mv.draw_rope(scn, (pts @ D.T + sh) @ R.T + pb, 1.0, 1.3, False)
            if kd in meta:
                D, sh = pods[kd][1], pods[kd][2]
                pts = mv.ab_map(mv.rope_path(mv.MB, mv.AB_MID, -st["ab"], ab=True))
                mv.draw_rope(scn, (pts @ D.T + sh) @ R.T + pb, 1.0, 1.3, False)
        img = Image.fromarray(renderer.render())
        raw = img
        img = mv.overlay(img, F, fi / mv.FPS, (np.array(lookat), dist, az, el), title, mv.THEMES["dark"])
        if at:
            img.save(out.parent / f"{out.stem}_{fi:05d}.jpg", quality=85)
        else:
            procs[0].stdin.write(img.tobytes())
            procs[1].stdin.write(raw.convert("RGB").tobytes())
        if fi % (mv.FPS * 10) == 0:
            print(f"  {out.name}: {fi / mv.FPS:5.1f}/{dur:.1f} s", flush=True)
    for pr in procs:
        pr.stdin.close()
        pr.wait()
    renderer.close()


if __name__ == "__main__":
    main()
