"""The tasks: what an episode is, what scenarios the bank holds, what each reward term measures, what success means.

  walk       flat ground, any direction: forward / sideways speed and turn rate commands (teleop's policy)
  terrain    a 7-8 m course straight ahead: flat, rough, slope, stairs up, stairs down, gaps, obstacles, mixed;
             10 difficulty levels each. Success = reach the goal. A blind policy: it feels the terrain
  rear       stand on the hind legs on command (command[0] = 1), back to four legs on 0
  handstand  stand on the front legs on command
  balance    stand on a board on a ball (a rola-bola with a sphere) and keep the board off the floor

Each task's reward is a weighted sum of named terms (WEIGHTS), every term logged on its own by train.py, and any
weight overridable from the command line (--w=track_lin=2,slip=-0.5). The sum is scaled by SCALE for learning.
"""
import math

import jax
import jax.numpy as jp
import numpy as np

import core as C

SCALE = 0.1


def _phys(r, n, hard=False):
    """Robot and world randomisation shared by every task: one draw per configuration."""
    f = lambda a, b: r.uniform(a, b, n)                                # noqa: E731
    return dict(fric=f(0.4, 1.25), mass_add=f(-0.5, 3.0 if hard else 2.0), strength=f(0.75, 1.05),
                kp_s=f(0.85, 1.15), kd_s=f(0.8, 1.2), w0_s=f(*C.hw.BATTERY_SAG))


def _push(r, n, vmax, p=0.7, t0=60, t1=360):
    a = r.uniform(-math.pi, math.pi, n)
    v = r.uniform(0.0, 1.0, n) * vmax * (r.random(n) < p)
    return dict(push_v=np.stack([np.cos(a), np.sin(a)], 1) * v[:, None], push_t=r.integers(t0, t1, n))


def _to_jax(bank):
    return {k: jp.asarray(v, jp.int32 if v.dtype.kind in "iu" else jp.float32) for k, v in bank.items()}


class Task:
    name = "task"
    extras = ()
    EP = 400
    WEIGHTS = {}
    N_PRIV_TASK = 6
    N_PRIV = 16 + N_PRIV_TASK
    clock = False
    obs_hist = C.N_HIST                          # frames in the observation: N_LONG for a distilled student

    def __init__(self, weights=None):
        self.w = dict(self.WEIGHTS)
        for k, v in (weights or {}).items():
            if k not in self.w:
                raise KeyError(f"{self.name} has no reward term {k!r}: {sorted(self.w)}")
            self.w[k] = float(v)

    def bind(self, L):
        self.L = L
        return self

    # --- what every task shares ----------------------------------------------------------------------------
    def mparams(self, cfg):
        L = self.L
        mp = dict(geom_friction=jp.asarray(L.fric0).at[:, 0].multiply(cfg["fric"]),
                  body_mass=jp.asarray(L.mass0).at[L.torso].add(cfg["mass_add"]))
        return mp

    def base_aux(self, cfg, i, key):
        k1, k2 = jax.random.split(key)
        return dict(cmd=jp.zeros(3), phase=jax.random.uniform(k1), clock_on=jp.bool_(self.clock), last_a=jp.zeros(12),
                    prev_target=jp.asarray(self.L.default), t=jp.int32(0), cfg=jp.int32(i), feet_prev=jp.zeros((4, 3)),
                    kp=jp.asarray(C.KP) * cfg["kp_s"], kd=jp.asarray(C.KD) * cfg["kd_s"], strength=cfg["strength"],
                    w0=cfg["w0_s"], sens=C.sensor_draw(k2), hist=jp.zeros((C.N_LONG, C.N_FRAME)),
                    cat=jp.float32(0.0))                              # train.py sets it in training worlds (--cat)

    def qpos0(self, p, yaw):
        q = jp.zeros(self.L.nq).at[0:3].set(p).at[3:7].set(C.yaw_quat(yaw))
        return q

    def ground(self, cfg, x, y):
        return jp.float32(0.0)

    def contacts(self, s, cfg):
        gz = jax.vmap(lambda f: self.ground(cfg, f[0], f[1]))(s["feet"])
        return s["feet"][:, 2] - gz < C.G.FOOT_R + 0.006

    def surface(self, s, cfg):
        """What the downward range and flow sensors look at: its height under the body, its body-frame velocity."""
        return self.ground(cfg, s["p"][0], s["p"][1]), jp.zeros(2)

    def observe(self, s, aux, cfg, key, first=False):
        gz, gv = self.surface(s, cfg)
        frame, sens = C.sense(self.L, s, aux, gz, key, gv)
        hist = jp.where(first, jp.tile(frame, (C.N_LONG, 1)), jp.concatenate([aux["hist"][1:], frame[None]], 0))
        return dict(aux, sens=sens, hist=hist), hist[-self.obs_hist:].ravel()

    def first_obs(self, s, aux, cfg, key):
        return self.observe(s, aux, cfg, key, first=True)

    def priv(self, s, aux, cfg, contact, extra):
        h = s["p"][2] - self.ground(cfg, s["p"][0], s["p"][1])
        return jp.concatenate([s["v"], jp.array([h - self.L.z_stand]), contact.astype(jp.float32),
                               jp.array([cfg["fric"], cfg["mass_add"], cfg["strength"], cfg["kp_s"], cfg["kd_s"], cfg["w0_s"],
                                         aux["sens"]["delay"] / 10.0, aux["sens"]["flow_scale"]]), extra])

    def extra(self, s, aux, cfg, feet=True):
        """The task's part of the privileged state (N_PRIV_TASK). feet=False: right after a reset, before the foot
        positions are updated."""
        return jp.zeros(self.N_PRIV_TASK)

    def priv_now(self, s, aux, cfg):
        """The privileged state at the first step of an episode (train.py: the critic, and a teacher's actor)."""
        return self.priv(s, aux, cfg, jp.ones(4, bool), self.extra(s, aux, cfg, feet=False))

    def begin(self, aux, cfg, qvel):
        """Before the physics of a step: a shove."""
        hit = aux["t"] == cfg["push_t"]
        return jp.where(hit, qvel.at[0:2].add(cfg["push_v"]), qvel), aux

    def common_terms(self, s, aux, a, tau, contact):
        fv = (s["feet"] - aux["feet_prev"]) / (C.DT * C.DECIM)
        slip = jp.sum(contact * jp.sum(fv[:, :2] ** 2, 1)) * (aux["t"] > 0)
        g = -s["R"][2, :]
        return dict(torque=jp.sum(tau ** 2), action_rate=jp.sum((a - aux["last_a"]) ** 2), slip=slip,
                    angvel=s["w"][0] ** 2 + s["w"][1] ** 2, tilt=g[0] ** 2 + g[1] ** 2, bounce=s["v_world"][2] ** 2)

    def finish(self, s, aux, cfg, a, raw, contact, fallen, success, trunc, extra, key, err, cut=False):
        """cut: a constraint ended the episode (train.py --cat): done, not a time-out, so no bootstrapping."""
        bad = ~jp.all(jp.isfinite(s["p"])) | ~jp.all(jp.isfinite(s["v"])) | ~jp.all(jp.isfinite(s["pos"]))
        fallen = fallen | bad
        raw = dict(raw, fall=fallen.astype(jp.float32), success=(success & ~fallen).astype(jp.float32))
        terms = {k: self.w[k] * raw[k] for k in self.w}
        r = sum(terms.values()) * SCALE
        aux = dict(aux, last_a=a, t=aux["t"] + 1, feet_prev=s["feet"])
        aux, obs = self.observe(s, aux, cfg, key)
        pv = self.priv(s, aux, cfg, contact, extra)
        r, obs, pv = jp.where(bad, -10.0 * SCALE, r), jp.where(bad, 0.0, obs), jp.where(bad, 0.0, pv)
        trunc = trunc & ~fallen & ~success
        done = fallen | success | trunc | cut
        trunc = trunc & ~cut
        terms = {k: jp.where(bad, 0.0, v) for k, v in terms.items()}
        return aux, obs, pv, r, terms, done, trunc, success & ~fallen, fallen, err


# =====================================================================================================================
class Walk(Task):
    """Flat ground, two 4 s commands and a shove; success = no fall and rms tracking errors under 0.2 m/s, 0.35 rad/s."""
    name = "walk"
    clock = True
    WEIGHTS = dict(track_lin=1.5, track_yaw=0.75, gait=0.5, air=1.0, bounce=-2.0, angvel=-0.05, tilt=-5.0, height=-20.0,
                   torque=-2e-4, action_rate=-0.01, slip=-0.2, idle_pose=-0.2, fall=-10.0, success=10.0)
    CMD = np.array([[-0.5, 1.2], [-0.4, 0.4], [-1.2, 1.2]])
    SWITCH, SETTLE = 200, 30

    def bank(self, n, seed=0):
        r = np.random.default_rng(seed)

        def cmds():
            c = r.uniform(self.CMD[:, 0], self.CMD[:, 1], (n, 3))
            c[r.random(n) < 0.12] = 0.0
            one = r.random(n) < 0.3
            c[one] *= np.eye(3)[r.integers(0, 3, n)[one]]
            c[np.abs(c) < 0.05] = 0.0
            return c
        b = dict(cmd_a=cmds(), cmd_b=cmds(), yaw0=r.uniform(-math.pi, math.pi, n), level=np.zeros(n, np.int32),
                 kind=np.zeros(n, np.int32), **_phys(r, n), **_push(r, n, 0.8))
        return _to_jax(b)

    def reset(self, cfg, i, key):
        L = self.L
        aux = self.base_aux(cfg, i, key)
        aux.update(cmd=cfg["cmd_a"], air=jp.zeros(4), e_lin=jp.float32(0), e_yaw=jp.float32(0), n_err=jp.float32(0))
        return self.qpos0(jp.array([0.0, 0.0, L.z0]), cfg["yaw0"]), jp.zeros(L.nv), aux, self.mparams(cfg)

    def end(self, s, aux, cfg, a, tau, key):
        L = self.L
        t = aux["t"] + 1
        cmd = aux["cmd"]
        contact = self.contacts(s, cfg)
        moving = jp.linalg.norm(cmd) > 0.0
        e_lin = (-s["v"][0] - cmd[0]) ** 2 + (s["v"][1] - cmd[1]) ** 2
        e_yaw = (s["w"][2] - cmd[2]) ** 2
        phase = (aux["phase"] + C.DT * C.DECIM / C.T_GAIT) % 1.0
        want = jp.where(moving, (phase + C.OFF) % 1.0 < 0.5, True)
        first = contact & (aux["air"] > 0.0)
        raw = self.common_terms(s, aux, a, tau, contact)
        raw.update(track_lin=jp.exp(-e_lin / 0.25), track_yaw=jp.exp(-e_yaw / 0.25), gait=jp.mean(contact == want),
                   air=jp.where(moving, jp.sum((aux["air"] - 0.5 * C.T_GAIT) * first), 0.0),
                   height=(s["p"][2] - L.z_stand) ** 2,
                   idle_pose=jp.where(moving, 0.0, jp.sum((s["pos"] - L.default) ** 2)))
        g = -s["R"][2, :]
        fallen = (g[2] > -0.5) | (s["p"][2] < 0.22)
        scored = (t % self.SWITCH) >= self.SETTLE
        e_l = aux["e_lin"] + jp.where(scored, e_lin, 0.0)
        e_y = aux["e_yaw"] + jp.where(scored, e_yaw, 0.0)
        n_e = aux["n_err"] + scored
        trunc = t >= self.EP
        rms_l, rms_y = jp.sqrt(e_l / jp.maximum(n_e, 1)), jp.sqrt(e_y / jp.maximum(n_e, 1))
        success = trunc & ~fallen & (rms_l < 0.2) & (rms_y < 0.35)
        aux = dict(aux, cmd=jp.where(t >= self.SWITCH, cfg["cmd_b"], cmd), phase=phase,
                   air=jp.where(contact, 0.0, aux["air"] + C.DT * C.DECIM), e_lin=e_l, e_yaw=e_y, n_err=n_e)
        return self.finish(s, aux, cfg, a, raw, contact, fallen, success, trunc, jp.zeros(self.N_PRIV_TASK), key,
                           jp.sqrt(e_lin))


# =====================================================================================================================
TERRAINS = ("flat", "rough", "slope", "stairs_up", "stairs_down", "gaps", "obstacles", "mixed")


def course(kind, level, r):
    """Rows of the course (N_ROWS): per row the centre along the forward direction s = -x, the top height of its
    two cells, and its tilt. Level 0..9 sets the difficulty. Returns boxes (N_BOX, 4): x, y, z centre, tilt."""
    x = level / 9.0
    # level 0 -> 9: rough +-0.5 -> 6 cm, slopes 2 -> 24 deg, steps 2 -> 16 cm, gaps 2 -> 27 cm, obstacles 2 -> 14 cm
    hmax = dict(rough=0.005 + 0.055 * x, slope=math.radians(2 + 22 * x), stairs=0.02 + 0.14 * x, gaps=0.02 + 0.25 * x,
                obstacles=0.02 + 0.12 * x)
    rows = []
    st = dict(s=-0.75, z=C.BASE)

    def put(top_l, top_r, phi=0.0):
        L_ = 2 * C.ROW_HALF * math.cos(phi)
        rows.append((st["s"] + L_ / 2, top_l, top_r, phi))
        st["s"] += L_

    def segment(kind, nrows):
        if kind == "flat":
            for _ in range(nrows):
                put(st["z"], st["z"])
        elif kind == "rough":
            a = hmax["rough"]
            for _ in range(nrows):
                put(st["z"] + r.uniform(-a, a), st["z"] + r.uniform(-a, a))
        elif kind == "slope":
            th = hmax["slope"]
            half = nrows // 2
            for j in range(nrows):
                phi = th if j < half else -th
                L_ = 2 * C.ROW_HALF
                zc = st["z"] + math.sin(phi) * L_ / 2
                put(zc, zc, phi)
                st["z"] += math.sin(phi) * L_
        elif kind in ("stairs_up", "stairs_down"):
            h = hmax["stairs"] * (1 if kind == "stairs_up" else -1)
            for j in range(nrows):
                if j % 2 == 0 and j > 0:
                    st["z"] += h
                put(st["z"], st["z"])
        elif kind == "gaps":
            g = hmax["gaps"]
            for j in range(nrows):
                if j % 3 == 0 and j > 0:
                    st["s"] += g
                put(st["z"], st["z"])
        elif kind == "obstacles":
            hm = hmax["obstacles"]
            for _ in range(nrows):
                bump = lambda: (r.random() < 0.35) * r.uniform(0.5, 1.0) * hm   # noqa: E731
                put(st["z"] + bump(), st["z"] + bump())

    n_start = 5                                                         # flat start: the robot spawns at s = 0
    if kind == "stairs_down":
        st["z"] = C.BASE + hmax["stairs"] * ((C.N_ROWS - n_start) // 2)
    segment("flat", n_start)
    left = C.N_ROWS - n_start
    if kind == "mixed":
        while left > 0:
            k = r.choice(["rough", "slope", "stairs_up", "stairs_down", "gaps", "obstacles"])
            if k == "stairs_down" and st["z"] - hmax["stairs"] * 3 < C.BASE:
                k = "stairs_up"
            nr = min(6, left)
            segment(k, nr)
            left -= nr
    else:
        segment(kind, left)
    boxes = []
    for s_c, zl, zr, phi in rows:
        for yc, zt in ((C.COL_HALF, zl), (-C.COL_HALF, zr)):
            boxes.append((-s_c - C.BOX_HALF_H * math.sin(phi), yc, zt - C.BOX_HALF_H * math.cos(phi), phi))
    return np.array(boxes, np.float32)


class Terrain(Task):
    """Walk the course straight ahead at the commanded speed (an operator steers: sideways and turn commands hold
    the robot on the course's centreline). Success = reach the goal, 85 % of speed x 8 s down the course.

    Privileged state (the critic's, and a teacher's input: train.py --teacher=1): a height scan of the course around
    the body, 10 points from 0.3 m behind to 1.05 m ahead x 3 across, and each foot's height over the ground under it.
    The robot has no sensor for either; a blind student learns to act like the teacher from what it feels (distill.py)."""
    name = "terrain"
    extras = ("terrain",)
    clock = True
    SCAN_FWD = np.linspace(-0.3, 1.05, 10)
    SCAN_SIDE = np.array([-0.12, 0.0, 0.12])
    N_PRIV_TASK = len(SCAN_FWD) * len(SCAN_SIDE) + 4
    N_PRIV = 16 + N_PRIV_TASK
    WEIGHTS = dict(track_lin=1.5, track_yaw=0.5, gait=0.3, air=1.0, bounce=-0.5, angvel=-0.05, tilt=-2.0, height=-5.0,
                   torque=-2e-4, action_rate=-0.01, slip=-0.2, fall=-10.0, success=10.0)

    def bank(self, n, seed=0):
        r = np.random.default_rng(seed)
        kind = np.arange(n) % len(TERRAINS)
        level = (np.arange(n) // len(TERRAINS)) % 10
        boxes = np.stack([course(TERRAINS[k], lv, r) for k, lv in zip(kind, level)])
        v = r.uniform(0.3, 0.9, n)
        b = dict(kind=kind.astype(np.int32), level=level.astype(np.int32), boxes=boxes, v_cmd=v, goal=0.85 * v * 8.0,
                 yaw0=r.uniform(-0.2, 0.2, n), y0=r.uniform(-0.05, 0.05, n), **_phys(r, n), **_push(r, n, 0.5, 0.5))
        return _to_jax(b)

    def ground(self, cfg, x, y):
        return C.ground_z(cfg["boxes"], x, y)

    def mocap(self, cfg):
        bx = cfg["boxes"]
        quat = jp.stack([jp.cos(bx[:, 3] / 2), jp.zeros(C.N_BOX), jp.sin(bx[:, 3] / 2), jp.zeros(C.N_BOX)], 1)
        return dict(mocap_pos=jp.zeros((C.N_BOX, 3)).at[self.L.terrain].set(bx[:, :3]),
                    mocap_quat=jp.zeros((C.N_BOX, 4)).at[self.L.terrain].set(quat))

    def reset(self, cfg, i, key):
        L = self.L
        aux = self.base_aux(cfg, i, key)
        aux.update(air=jp.zeros(4))
        z = self.ground(cfg, 0.0, cfg["y0"]) + L.z0
        return self.qpos0(jp.array([0.0, cfg["y0"], z]), cfg["yaw0"]), jp.zeros(L.nv), aux, self.mparams(cfg) | self.mocap(cfg)

    def extra(self, s, aux, cfg, feet=True):
        yaw = jp.arctan2(s["R"][1, 0], s["R"][0, 0])
        fwd, side = -jp.array([jp.cos(yaw), jp.sin(yaw)]), jp.array([-jp.sin(yaw), jp.cos(yaw)])
        pts = (s["p"][:2] + jp.asarray(self.SCAN_FWD)[:, None, None] * fwd + jp.asarray(self.SCAN_SIDE)[None, :, None] * side).reshape(-1, 2)
        gz = jax.vmap(lambda q: self.ground(cfg, q[0], q[1]))(pts)
        scan = (s["p"][2] - gz - self.L.z_stand) * 5.0
        fz = jax.vmap(lambda f: self.ground(cfg, f[0], f[1]))(s["feet"])
        clear = jp.clip((s["feet"][:, 2] - fz - C.G.FOOT_R) * 10.0, -1.0, 3.0) * feet
        return jp.concatenate([scan, clear])

    def end(self, s, aux, cfg, a, tau, key):
        L = self.L
        t = aux["t"] + 1
        cmd = aux["cmd"]
        contact = self.contacts(s, cfg)
        e_lin = (-s["v"][0] - cmd[0]) ** 2 + (s["v"][1] - cmd[1]) ** 2
        e_yaw = (s["w"][2] - cmd[2]) ** 2
        phase = (aux["phase"] + C.DT * C.DECIM / C.T_GAIT) % 1.0
        want = (phase + C.OFF) % 1.0 < 0.5
        first = contact & (aux["air"] > 0.0)
        gz = self.ground(cfg, s["p"][0], s["p"][1])
        raw = self.common_terms(s, aux, a, tau, contact)
        raw.update(track_lin=jp.exp(-e_lin / 0.25), track_yaw=jp.exp(-e_yaw / 0.25), gait=jp.mean(contact == want),
                   air=jp.sum((aux["air"] - 0.5 * C.T_GAIT) * first), height=(s["p"][2] - gz - L.z_stand) ** 2)
        g = -s["R"][2, :]
        fallen = (g[2] > -0.5) | (s["p"][2] - gz < 0.15) | (s["p"][2] < C.BASE * 0.5)
        progress = -s["p"][0]
        success = progress >= cfg["goal"]
        trunc = t >= self.EP
        yaw = jp.arctan2(s["R"][1, 0], s["R"][0, 0])
        nxt = jp.array([jp.minimum(cmd[0] + 0.04, cfg["v_cmd"]), jp.clip(-1.5 * s["p"][1], -0.3, 0.3), jp.clip(-2.0 * yaw, -1.0, 1.0)])
        aux = dict(aux, cmd=nxt, phase=phase, air=jp.where(contact, 0.0, aux["air"] + C.DT * C.DECIM))
        return self.finish(s, aux, cfg, a, raw, contact, fallen, success, trunc, self.extra(s, aux, cfg), key,
                           progress / cfg["goal"])


# =====================================================================================================================
class Pose(Task):
    """Stand on one pair of legs on command. command[0] switches 0 -> 1 between 0.5 and 1.5 s; success = the body's
    forward axis at least 50 deg up (rear) or down (handstand) for 80 % of the last 2 s, without falling. The body
    touching the ground counts as a fall (a first try leaned the handstand on the body's front edge)."""
    EP = 300
    UP = 1.0                                     # +1 rear (nose up), -1 handstand (nose down)
    LIFT = None
    WEIGHTS = dict(pose=2.5, feet=0.5, roll=-3.0, drift=-1.0, yaw_rate=-0.1, torque=-1e-4, action_rate=-0.01,
                   slip=-0.1, fall=-10.0, success=10.0, cut=-1.0)
    # constraints as terminations (train.py --cat=p_max): 1.5 s after the command the body must stay above the bar,
    # and no motor past 85 % of its limit. Violation 0..1 each; the episode ends with probability p_max * the worst
    TAU_OK, POSE_AFTER = 0.85, 75
    F_TARGET, F_OK, HOLD = 0.9, 0.766, 100
    SIGMA = 0.15                                 # width of the pose kernel exp(-(f_z - target)^2 / SIGMA)

    def bank(self, n, seed=0):
        r = np.random.default_rng(seed)
        level = np.arange(n) % 10
        x = (level + 1) / 10.0
        a = r.uniform(-math.pi, math.pi, n)
        b = dict(level=level.astype(np.int32), kind=np.zeros(n, np.int32), t_on=r.integers(25, 75, n),
                 yaw0=r.uniform(-math.pi, math.pi, n), **_phys(r, n), push_t=r.integers(100, 280, n),
                 push_v=np.stack([np.cos(a), np.sin(a)], 1) * (0.6 * x * r.random(n))[:, None])   # harder: bigger shoves
        b["strength"] = 1.05 - 0.3 * x * r.random(n)                       # ... and weaker motors
        return _to_jax(b)

    def reset(self, cfg, i, key):
        L = self.L
        aux = self.base_aux(cfg, i, key)
        aux.update(hold=jp.int32(0))
        return self.qpos0(jp.array([0.0, 0.0, L.z0]), cfg["yaw0"]), jp.zeros(L.nv), aux, self.mparams(cfg)

    def end(self, s, aux, cfg, a, tau, key):
        t = aux["t"] + 1
        mode = aux["cmd"][0]
        contact = self.contacts(s, cfg)
        f_z = -s["R"][2, 0] * self.UP                                      # forward axis up (rear) / down (handstand)
        lift = np.asarray(self.LIFT)
        want = jp.ones(4, bool).at[lift].set(mode < 0.5)
        raw = self.common_terms(s, aux, a, tau, contact)
        raw.update(pose=jp.exp(-(f_z - mode * self.F_TARGET) ** 2 / self.SIGMA), feet=jp.mean(contact == want),
                   margin=mode * jp.clip((f_z - self.F_OK) / 0.1, 0.0, 1.0),
                   roll=s["R"][2, 1] ** 2, drift=jp.sum(s["v_world"][:2] ** 2), yaw_rate=s["w"][2] ** 2)
        low = jp.min(s["p"][2] + jp.asarray(self.L.torso_corners) @ s["R"][2, :])   # lowest corner of the body
        fallen = (jp.abs(s["R"][2, 1]) > 0.7) | (s["p"][2] < 0.12) | (f_z < -0.6) | (low < 0.015)
        hold = aux["hold"] + ((t > self.EP - self.HOLD) & (f_z >= self.F_OK))
        trunc = t >= self.EP
        success = trunc & ~fallen & (hold >= 0.8 * self.HOLD)
        v_pose = (t > cfg["t_on"] + self.POSE_AFTER) * jp.clip((self.F_OK + 0.02 - f_z) / 0.2, 0.0, 1.0)
        v_tau = jp.clip((jp.max(tau / (C.M.TAU * aux["strength"])) - self.TAU_OK) / 0.15, 0.0, 1.0)
        key, kc = jax.random.split(key)
        cut = (jax.random.uniform(kc) < aux["cat"] * jp.maximum(v_pose, v_tau)) & ~fallen & ~success & ~trunc
        raw["cut"] = cut.astype(jp.float32)
        aux = dict(aux, hold=hold, cmd=jp.array([jp.where(t >= cfg["t_on"], 1.0, 0.0), 0.0, 0.0]))
        extra = jp.array([f_z, mode, cfg["level"] / 10.0, 0.0, 0.0, 0.0])
        return self.finish(s, aux, cfg, a, raw, contact, fallen, success, trunc, extra, key, jp.abs(f_z - mode * self.F_TARGET), cut)


class Rear(Pose):
    name = "rear"
    UP = 1.0
    LIFT = C.FRONT                                                       # front feet come off the ground


class Handstand(Pose):
    """The first 300 M-step run held the body at ~45-50 deg nose-down, right at the 50 deg bar (14 % success): the
    0.15-wide kernel paid almost as much at 45 deg as at 64. Narrower kernel aimed at ~60 deg (0.87), and a smooth
    'margin' term for being above the bar (0 at 50 deg, full at ~58 deg) instead of a step that invites hovering."""
    name = "handstand"
    UP = -1.0
    LIFT = C.HIND
    F_TARGET, SIGMA = 0.87, 0.05
    WEIGHTS = dict(Pose.WEIGHTS, margin=1.0)


# =====================================================================================================================
class Balance(Task):
    """Stand on a board resting on a ball for 8 s. Fail if the board touches the floor or the robot falls or steps
    off. Difficulty: how freely the ball rolls (damping on its linear motion, high at level 0, zero at 9), shoves."""
    name = "balance"
    extras = ("board",)
    WEIGHTS = dict(alive=1.0, board_level=-3.0, upright=-2.0, ball_drift=-0.5, feet_on=0.5, rel_vel=-0.5, angvel=-0.05,
                   torque=-1e-4, action_rate=-0.01, fall=-10.0, success=10.0)

    def bank(self, n, seed=0):
        r = np.random.default_rng(seed)
        level = np.arange(n) % 10
        x = level / 9.0
        b = dict(level=level.astype(np.int32), kind=np.zeros(n, np.int32), damp=200.0 * (1 - x) ** 2,
                 ball_off=r.uniform(-1, 1, (n, 2)) * 0.02 * x[:, None], yaw0=r.uniform(-0.3, 0.3, n), **_phys(r, n))
        a = r.uniform(-math.pi, math.pi, n)
        b.update(push_t=r.integers(100, 300, n), push_v=np.stack([np.cos(a), np.sin(a)], 1) * (0.3 * x * r.random(n))[:, None])
        return _to_jax(b)

    def board_top(self, s, x, y):
        n = s["board_R"][:, 2]
        c = s["board_p"]
        # plane through the top face: n . (p - c) = h  ->  z
        return c[2] + (C.BOARD_HALF[2] - n[0] * (x - c[0]) - n[1] * (y - c[1])) / jp.maximum(n[2], 0.3)

    def surface(self, s, cfg):
        # the sensors look at the board while the body is over it, else at the floor (a first version always used
        # the floor, through the board: range 0.2 m too long and flow relative to the world, not the board)
        rel = s["board_R"].T @ (s["p"] - s["board_p"])
        over = (jp.abs(rel[0]) < C.BOARD_HALF[0]) & (jp.abs(rel[1]) < C.BOARD_HALF[1])
        gz = jp.where(over, self.board_top(s, s["p"][0], s["p"][1]), 0.0)
        gv = jp.where(over, (s["R"].T @ s["board_v"])[:2], jp.zeros(2))
        return gz, gv

    def contacts(self, s, cfg):
        gz = jax.vmap(lambda f: self.board_top(s, f[0], f[1]))(s["feet"])
        return jp.abs(s["feet"][:, 2] - gz) < C.G.FOOT_R + 0.006

    def mparams(self, cfg):
        L = self.L
        mp = super().mparams(cfg)
        mp["dof_damping"] = jp.asarray(L.dof_damping0).at[L.ball_v:L.ball_v + 3].set(cfg["damp"])
        return mp

    def reset(self, cfg, i, key):
        L = self.L
        aux = self.base_aux(cfg, i, key)
        top = 2 * C.BALL_R + 2 * C.BOARD_HALF[2]
        q = self.qpos0(jp.array([0.0, 0.0, top + L.z0]), cfg["yaw0"])
        q = q.at[L.ball_q:L.ball_q + 3].set(jp.array([cfg["ball_off"][0], cfg["ball_off"][1], C.BALL_R])).at[L.ball_q + 3].set(1.0)
        q = q.at[L.board_q:L.board_q + 3].set(jp.array([0.0, 0.0, 2 * C.BALL_R + C.BOARD_HALF[2]])).at[L.board_q + 3].set(1.0)
        return q, jp.zeros(L.nv), aux, self.mparams(cfg)

    def end(self, s, aux, cfg, a, tau, key):
        t = aux["t"] + 1
        contact = self.contacts(s, cfg)
        n = s["board_R"][:, 2]
        hx, hy, hz = C.BOARD_HALF
        corners = jp.array([[sx * hx, sy * hy, -hz] for sx in (-1, 1) for sy in (-1, 1)])
        low = jp.min(s["board_p"][2] + corners @ s["board_R"][2, :])
        g = -s["R"][2, :]
        rel = s["p"] - s["board_p"]
        raw = self.common_terms(s, aux, a, tau, contact)
        raw.update(alive=jp.float32(1.0), board_level=n[0] ** 2 + n[1] ** 2, upright=g[0] ** 2 + g[1] ** 2,
                   ball_drift=jp.sum(s["ball_p"][:2] ** 2), feet_on=jp.mean(contact),
                   rel_vel=jp.sum((s["v_world"][:2] - s["board_v"][:2]) ** 2))
        off = (jp.abs(rel[0]) > hx + 0.05) | (jp.abs(rel[1]) > hy + 0.05)
        fallen = (low < 0.004) | (g[2] > -0.6) | (rel[2] < 0.15) | off
        trunc = t >= self.EP
        success = trunc & ~fallen
        extra = jp.concatenate([n[:2] * 5.0, s["ball_p"][:2] * 5.0, rel[:2] * 5.0])
        return self.finish(s, aux, cfg, a, raw, contact, fallen, success, trunc, extra, key, jp.sqrt(n[0] ** 2 + n[1] ** 2))


TASKS = dict(walk=Walk, terrain=Terrain, rear=Rear, handstand=Handstand, balance=Balance)
