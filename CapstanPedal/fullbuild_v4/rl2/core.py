"""The v4 robot for learning, shared by the GPU trainer and the CPU checks.

  build_spec(kind, extras)  the MuJoCo model: "physics" (GPU: meshes dropped, rotors welded) or "full" (every mesh,
                            rotors on their joints), plus the task's extras: a terrain course of boxes, a ball and
                            a board
  Layout                    where things are in a compiled model (joint, site, geom and body indices)
  torque()                  the PD loop and motor model (numpy or jax)
  targets()                 action -> output targets inside the firmware's limits
  sense()                   one observation frame as the robot's sensors give it (hw.py)
  ground_z()                height of the terrain course under a point

The robot's forward is -X: the "F" pods (at +X) are its hind legs.
"""
import math
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np

import hw

HERE = Path(__file__).resolve().parent
# The build whose robot this trains: by default the folder rl2/ sits in (fullbuild_v4). Its mujoco/model.py builds the
# model from its urdf/; its mujoco/gait.py gives the leg geometry and limits. CAPSTAN_BUILD=<path> points at another
# build. What a build must provide: README.md, "A new build version"; check it with check_robot.py.
BUILD = Path(os.environ.get("CAPSTAN_BUILD", HERE.parent)).resolve()
TAG = BUILD.name.replace("fullbuild_", "")                         # "v4": film names, ../video/<TAG>_rl2_*.mp4
VIDEO = BUILD / "video"
sys.path.insert(0, str(BUILD / "mujoco"))
import gait as G                                                     # noqa: E402
import model as M                                                    # noqa: E402

DT, DECIM = hw.PHYS_DT, round(1.0 / hw.POLICY_HZ / hw.PHYS_DT)      # 500 Hz physics, 10 steps per policy step
KP = np.array([80.0, 60.0, 60.0] * 4)          # N*m/rad per output: ab/ad, B, A
KD = np.array([1.5, 2.0, 2.0] * 4)
ACT_SCALE = np.array([0.2, 0.6, 0.6] * 4)      # rad of target offset per unit action
H_NOM = G.H_STAND - 0.01
T_GAIT = 0.30
OFF = np.array([0.5, 0.0, 0.0, 0.5])           # trot clock offset per pod (LF, LR, RF, RR)
HIND, FRONT = np.array([0, 2]), np.array([1, 3])
N_FRAME = 51                                    # one observation frame (sense())
N_HIST = 3                                      # frames stacked into the policy's input
N_OBS = N_FRAME * N_HIST
N_LONG = 50                                     # frames kept (1 s): what a distilled student reads (distill.py)
CMD_SCALE = np.array([2.0, 2.0, 0.25])

# terrain course: rows of boxes along the robot's forward direction (-X), two columns, tilted for slopes
N_ROWS, ROW_HALF, COL_HALF, BOX_HALF_H = 32, 0.125, 0.3, 0.5
N_BOX = 2 * N_ROWS
BASE = 0.3                                      # course top above the floor: falling off the side is a fall
# balance board on a ball
BALL_R = 0.10
BOARD_HALF = (0.45, 0.28, 0.012)
BOARD_MASS, BALL_MASS = 2.0, 1.5


def build_spec(kind="physics", extras=()):
    spec = M.build_spec()
    if kind == "physics":
        for g in list(spec.geoms):
            if g.type == mujoco.mjtGeom.mjGEOM_MESH:
                spec.delete(g)
        for me in list(spec.meshes):
            spec.delete(me)
        for e in list(spec.equalities):
            if e.type == mujoco.mjtEq.mjEQ_JOINT:
                spec.delete(e)
        for j in list(spec.joints):
            if "_rotor" in j.name:
                spec.delete(j)
    if kind == "full":
        add_sensor_boards(spec)
    wb = spec.worldbody
    if "terrain" in extras:
        # mocap bodies: their poses live in the simulation data, so every world gets its own course. Collision bit 2:
        # the robot collides with them, they do not collide with each other or the floor
        for g in spec.geoms:
            if g.contype or g.conaffinity:
                if g.name != "floor":
                    g.conaffinity = g.conaffinity | 2
        for k in range(N_BOX):
            b = wb.add_body(name=f"terrain{k}", mocap=True, pos=[0.0, 0.0, -2.0 - k])
            b.add_geom(name=f"terrain{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[ROW_HALF, COL_HALF, BOX_HALF_H],
                       rgba=[0.55, 0.57, 0.6, 1.0] if k % 2 else [0.48, 0.5, 0.54, 1.0], friction=[1.0, 0.005, 0.0001],
                       contype=2, conaffinity=0)
    if "board" in extras:
        ball = wb.add_body(name="ball", pos=[0.0, 0.0, BALL_R])
        ball.add_freejoint(name="ball")
        ball.add_geom(name="ball", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_R], mass=BALL_MASS,
                      rgba=[0.85, 0.35, 0.2, 1.0], friction=[1.0, 0.005, 0.0001])
        board = wb.add_body(name="board", pos=[0.0, 0.0, 2 * BALL_R + BOARD_HALF[2]])
        board.add_freejoint(name="board")
        board.add_geom(name="board", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(BOARD_HALF), mass=BOARD_MASS,
                       rgba=[0.75, 0.6, 0.4, 1.0], friction=[1.0, 0.005, 0.0001])
    return spec


def add_sensor_boards(spec):
    """The new ground-speed sensors (../BOM.md "Sensing") as visual-only boards under the bottom plate: the PMW3901
    flow breakout and the VL53L1X range breakout, lenses down. No collisions; the torso's inertia comes from the URDF."""
    torso = spec.body("torso")
    box = [g for g in torso.geoms if g.type == mujoco.mjtGeom.mjGEOM_BOX][0]
    z = box.pos[2] - box.size[2] - 0.006                                # 6 mm standoffs
    for name, x, size, rgba in (("flow_board", 0.02, [0.0128, 0.0128, 0.0008], [0.1, 0.45, 0.2, 1]),
                                ("tof_board", -0.02, [0.0128, 0.0102, 0.0008], [0.1, 0.25, 0.55, 1])):
        torso.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=[x, 0.0, z], rgba=rgba,
                       contype=0, conaffinity=0, group=1, mass=0.0)
        torso.add_geom(name=name + "_lens", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.004, 0.002],
                       pos=[x, 0.0, z - 0.0025], rgba=[0.05, 0.05, 0.05, 1], contype=0, conaffinity=0, group=1, mass=0.0)


class Layout:
    """Indices into one compiled model. Plain numpy / Python numbers, so jitted code bakes them in."""

    def __init__(self, m, extras=()):
        self.m = m
        legs = [G.Leg(m, p) for p in M.PODS]
        self.qa = np.array([lg.ja for lg in legs]).ravel()
        self.va = np.array([lg.jv for lg in legs]).ravel()
        self.act = np.array([lg.act for lg in legs]).ravel()
        self.foot_site = np.array([m.site(f"foot_{p}").id for p in M.PODS])
        self.torso = m.body("torso").id
        q = np.array([lg.ik(lg.nominal(H_NOM)) for lg in legs])
        self.default = np.stack([q[:, 0], q[:, 1], q[:, 1] + q[:, 2]], 1).ravel().astype(np.float32)
        self.z_stand = float(G.FOOT_R + H_NOM - legs[0].hip_z)        # torso origin over the ground, standing
        self.z0 = float(G.FOOT_R + G.H_STAND - legs[0].hip_z + 0.005)    # reset height: joints at zero
        tg = [i for i in range(m.ngeom) if m.geom_bodyid[i] == self.torso and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
              and m.geom_contype[i]]
        self.sensor_dz = -float(m.geom_size[tg[0], 2]) + float(m.geom_pos[tg[0], 2]) if tg else -0.04
        self.tof_nom = self.z_stand + self.sensor_dz
        # the torso box's eight corners in the body frame: a pose task must not rest the body on the ground
        half, off = (m.geom_size[tg[0]], m.geom_pos[tg[0]]) if tg else (np.array([0.25, 0.12, 0.05]), np.zeros(3))
        self.torso_corners = np.array([off + half * np.array([sx, sy, sz]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                                      np.float32)
        self.isA = np.arange(12) % 3 == 2
        self.nq, self.nv = m.nq, m.nv
        self.fric0 = m.geom_friction.copy()
        self.mass0 = m.body_mass.copy()
        self.terrain = np.array([m.body_mocapid[m.body(f"terrain{k}").id] for k in range(N_BOX)]) if "terrain" in extras else None
        if "board" in extras:
            self.ball_q, self.ball_v = int(m.jnt_qposadr[m.joint("ball").id]), int(m.jnt_dofadr[m.joint("ball").id])
            self.board_q, self.board_v = int(m.jnt_qposadr[m.joint("board").id]), int(m.jnt_dofadr[m.joint("board").id])
            self.dof_damping0 = m.dof_damping.copy()


# --- math (jax) ---------------------------------------------------------------------------------------------------
def quat_mat(q):
    q = q / jp.linalg.norm(q)
    w, x, y, z = q
    return jp.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def yaw_quat(yaw):
    return jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])


def state(L, qpos, qvel, site_xpos):
    """What the task code reads, from either an mjx Data or an MjData's arrays."""
    R = quat_mat(qpos[3:7])
    pos, vel = qpos[L.qa], qvel[L.va]
    s = dict(p=qpos[0:3], R=R, v_world=qvel[0:3], v=R.T @ qvel[0:3], w=qvel[3:6],
             pos=pos.at[2::3].add(pos[1::3]), vel=vel.at[2::3].add(vel[1::3]), feet=site_xpos[L.foot_site])
    if hasattr(L, "ball_q"):
        s.update(ball_p=qpos[L.ball_q:L.ball_q + 3], ball_v=qvel[L.ball_v:L.ball_v + 3],
                 board_p=qpos[L.board_q:L.board_q + 3], board_R=quat_mat(qpos[L.board_q + 3:L.board_q + 7]),
                 board_v=qvel[L.board_v:L.board_v + 3])
    return s


def ground_z(boxes, x, y):
    """Top of the terrain course at (x, y): the highest box top over the point, or the floor (0). boxes (K, 4):
    centre x, y, z and the tilt about y."""
    xc, yc, zc, ph = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    c, s = jp.cos(ph), jp.sin(ph)
    u = (x - (xc + BOX_HALF_H * s)) / c
    z = zc + BOX_HALF_H * c - u * s
    inside = (jp.abs(u) <= ROW_HALF) & (jp.abs(y - yc) <= COL_HALF)
    return jp.maximum(jp.max(jp.where(inside, z, -1.0)), 0.0)


# --- actuation (numpy or jax: xp) -----------------------------------------------------------------------------------
def targets(xp, default, a):
    t = (default + a * ACT_SCALE).reshape(4, 3)
    ab = xp.clip(t[:, 0], -G.ABAD_LIM, G.ABAD_LIM)
    b = xp.clip(t[:, 1], -G.OUT_LIM, G.OUT_LIM)
    a_ = xp.clip(t[:, 2], -G.OUT_LIM, G.OUT_LIM)
    a_ = b + xp.clip(a_ - b, -G.KNEE_SOFT, G.KNEE_SOFT)
    return xp.stack([ab, b, a_], 1).ravel()


def torque(xp, pos, vel, target, kp, kd, strength, w0_scale, isA):
    """PD at the output, limited by the motor: TAU to half the no-load speed (which sags with the battery),
    falling to zero at it; output A limited to A_PUSH pushing the coupler."""
    tau = kp * (target - pos) - kd * vel
    lim = M.TAU * strength * xp.clip(2.0 * (1.0 - xp.abs(vel) / (M.W0 * w0_scale)), 0.0, 1.0)
    hi = xp.minimum(lim, xp.where(isA, M.A_PUSH * strength, xp.inf))
    return xp.minimum(xp.maximum(tau, -lim), hi)


# --- sensing (jax) ---------------------------------------------------------------------------------------------------
def sensor_draw(key):
    """Per-episode sensor errors."""
    k = jax.random.split(key, 6)
    u = lambda kk, shape, lo, hi: jax.random.uniform(kk, shape, minval=lo, maxval=hi)   # noqa: E731
    return dict(gyro_bias=u(k[0], (3,), -hw.GYRO_BIAS, hw.GYRO_BIAS), tilt_bias=u(k[1], (2,), -hw.TILT_BIAS, hw.TILT_BIAS),
                enc_off=u(k[2], (12,), -hw.ENC_OFFSET, hw.ENC_OFFSET), flow_scale=u(k[3], (), -hw.FLOW_SCALE, hw.FLOW_SCALE),
                flow_drop=jp.where(jax.random.uniform(jax.random.fold_in(k[4], 1)) < hw.FLOW_DEAD, 1.0, u(k[4], (), *hw.FLOW_DROPOUT)), flow_last=jp.zeros(2), v_prev=jp.zeros(2),
                delay=jax.random.randint(k[5], (), round(hw.DELAY_MS[0] / 1000 / DT), round(hw.DELAY_MS[1] / 1000 / DT) + 1))


def sense(L, s, aux, ground, key, ground_v=None):
    """One observation frame (N_FRAME) as the hardware delivers it, and the sensor state for the next step. ground:
    height of the surface under the sensors; ground_v: that surface's velocity in the body frame (x, y), since the
    flow sensor measures motion relative to what it looks at (the board, in the balance task)."""
    k = jax.random.split(key, 8)
    nrm = lambda kk, shape, sd: jax.random.normal(kk, shape) * sd   # noqa: E731
    sn = aux["sens"]
    gyro = s["w"] + sn["gyro_bias"] + nrm(k[0], (3,), hw.GYRO_NOISE)
    g = -s["R"][2, :]
    g = g + jp.concatenate([sn["tilt_bias"] + nrm(k[1], (2,), hw.TILT_NOISE), jp.zeros(1)])
    g = g / jp.linalg.norm(g)
    # range: from the bottom plate straight down the body's -z, to the ground under the torso
    cos_t = jp.maximum(-g[2], 0.3)
    h_true = (s["p"][2] + L.sensor_dz - ground) / cos_t
    tof = jp.clip(h_true * (1 + nrm(k[2], (), hw.TOF_REL)) + nrm(k[3], (), hw.TOF_NOISE), *hw.TOF_RANGE)
    # flow: body-frame ground speed, one step old, scaled by the height error, noisy, sometimes invalid
    ok = (jax.random.uniform(k[4]) > sn["flow_drop"]) & (h_true > hw.FLOW_MIN_H) & (h_true < hw.FLOW_MAX_H)
    flow = jp.where(ok, sn["v_prev"] * (1 + sn["flow_scale"]) + nrm(k[5], (2,), hw.FLOW_NOISE), sn["flow_last"])
    pos = jp.round((s["pos"] + sn["enc_off"] + nrm(k[6], (12,), hw.ENC_NOISE)) / hw.ENC_QUANT) * hw.ENC_QUANT
    vel = s["vel"] + nrm(k[7], (12,), hw.VEL_NOISE)
    ph = 2 * jp.pi * aux["phase"]
    clock = jp.where(aux["clock_on"], jp.array([jp.sin(ph), jp.cos(ph)]), jp.zeros(2))
    frame = jp.concatenate([gyro * 0.25, g, flow * 2.0, ok[None].astype(jp.float32), ((tof - L.tof_nom) * 5.0)[None],
                            aux["cmd"] * CMD_SCALE, pos - L.default, vel * 0.05, aux["last_a"], clock])
    sn = dict(sn, flow_last=flow, v_prev=s["v"][:2] - (jp.zeros(2) if ground_v is None else ground_v))
    return frame, sn
