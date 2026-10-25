"""The v4 quadruped for learning: the URDF model (../mujoco/model.py) with a policy-rate interface and no gym.

Control at 50 Hz: the policy gives a target per drive OUTPUT (ab/ad, output B = hip, output A = the crank: the
shin's absolute angle), as offsets from the standing pose; a PD loop at 500 Hz turns them into torques, clipped
to the motor model of gait.py: 31.5 N*m to half the no-load speed, falling linearly to zero at W0, and output A
limited to 25 N*m in the direction that pushes the coupler. Targets are first clamped to the robot's software
limits (gait.Leg.clamp: ab/ad +-12, outputs +-88, knee +-52 deg), as the robot's firmware would.
The robot's forward is -X (the "F" pods, at +X, are its hind legs).

Used by envs.py (training) and by play.py / teleop.py (a trained policy, through policy.py).
"""
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mujoco"))
import gait as G                                                     # noqa: E402
import model as M                                                    # noqa: E402

DT, DECIM = 0.002, 10                          # physics step, physics steps per policy step (50 Hz)
KP = np.array([80.0, 60.0, 60.0] * 4)          # N*m/rad per output: ab/ad, B, A
KD = np.array([1.5, 2.0, 2.0] * 4)             # N*m*s/rad
ACT_SCALE = np.array([0.2, 0.6, 0.6] * 4)      # rad of target offset per unit action
H_NOM = G.H_STAND - 0.01                       # hip to foot centre standing (gait.py's default)
N_OBS = 50
T_GAIT = 0.30                                  # s per trot cycle (the clock in the observation)
OFF = np.array([0.5, 0.0, 0.0, 0.5])           # clock offset per pod (LF, LR, RF, RR): diagonal pairs, as gait.py
OK_LIN, OK_YAW = 0.2, 0.35                     # an episode succeeds: rms velocity / yaw-rate errors under these
SETTLE = 30                                    # policy steps after each new command not scored
LIMIT = mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT
# Task configuration ranges (mjx_walk.sample_bank for training, play.py eval --bank=): commands (fwd, side, yaw),
# floor friction scale, torso mass added (kg), motor strength, PD gain scales, shove speed (m/s) and its chance.
# "hard" reaches past what the robot can always do (fast commands, weak motors on ice, big shoves), so much of
# its bank is hard or hopeless: the case Success-Guided Sampling is for.
BANKS = dict(
    base=dict(cmd=[[-0.5, 1.2], [-0.4, 0.4], [-1.2, 1.2]], fric=[0.4, 1.25], mass=[-0.5, 2.0], strength=[0.8, 1.05],
              kp=[0.85, 1.15], kd=[0.8, 1.2], push=[0.0, 0.8], p_push=0.7),
    hard=dict(cmd=[[-0.8, 1.6], [-0.6, 0.6], [-2.0, 2.0]], fric=[0.25, 1.25], mass=[-1.0, 4.0], strength=[0.65, 1.05],
              kp=[0.75, 1.25], kd=[0.7, 1.3], push=[0.0, 1.6], p_push=0.85))


class Quad:
    def __init__(self, floor_friction=1.0):
        self.m = M.build(floor_friction=floor_friction)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.legs = [G.Leg(m, p) for p in M.PODS]
        self.qa = np.array([lg.ja for lg in self.legs]).ravel()       # qpos of abad, hip, knee per pod
        self.va = np.array([lg.jv for lg in self.legs]).ravel()
        self.act = np.array([lg.act for lg in self.legs]).ravel()     # mC, mB, mA per pod
        self.feet = np.array([m.geom(f"foot_{p}").id for p in M.PODS])
        self.foot_site = np.array([m.site(f"foot_{p}").id for p in M.PODS])
        self.floor = m.geom("floor").id
        robot = np.array([m.geom_bodyid[i] != 0 and m.geom_contype[i] for i in range(m.ngeom)])
        self.bad_geoms = np.flatnonzero(robot & ~np.isin(np.arange(m.ngeom), self.feet))
        self.torso = m.body("torso").id
        # the standing pose: each foot h below its hip, as outputs (ab/ad, B, A)
        q = np.array([lg.ik(lg.nominal(H_NOM)) for lg in self.legs])
        self.default = np.stack([q[:, 0], q[:, 1], q[:, 1] + q[:, 2]], 1).ravel()
        self.z_stand = G.FOOT_R + H_NOM - self.legs[0].hip_z            # torso origin height, standing
        self.kp, self.kd = KP.copy(), KD.copy()
        self.set_strength(1.0)
        self.last_a = np.zeros(12)
        self.cmd = np.zeros(3)                                          # forward m/s, left m/s, yaw rad/s
        self.phase = 0.0                                                # gait clock, 0..1

    # --- state ---------------------------------------------------------------------------------------
    def outputs(self):
        pos, vel = self.d.qpos[self.qa], self.d.qvel[self.va]
        pos[2::3] += pos[1::3]                                         # output A = hip + knee
        vel[2::3] += vel[1::3]
        return pos, vel

    def base(self):
        d = self.d
        R = d.xmat[self.torso].reshape(3, 3)
        v_body = R.T @ d.qvel[0:3]
        w_body = d.qvel[3:6].copy()                                    # the free joint's angular velocity: body frame
        grav = R[2, :] * -1.0                                          # gravity direction in the body frame
        return R, v_body, w_body, grav

    def foot_contacts(self):
        """(4,) bool feet on the floor, and whether anything else of the robot touches it."""
        n = self.d.ncon
        if n == 0:
            return np.zeros(4, bool), False
        g = self.d.contact.geom[:n]
        other = np.where(g[:, 0] == self.floor, g[:, 1], np.where(g[:, 1] == self.floor, g[:, 0], -1))
        return np.isin(self.feet, other), bool(np.isin(other, self.bad_geoms).any())

    def foot_vel(self):
        """(4, 3) world velocity of each foot centre."""
        v = np.zeros((4, 3))
        tmp = np.zeros(6)
        for i, s in enumerate(self.foot_site):
            mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_SITE, s, tmp, 0)
            v[i] = tmp[3:]
        return v

    def obs(self, noise=None):
        R, v, w, grav = self.base()
        pos, vel = self.outputs()
        o = np.concatenate([v * 2.0, w * 0.25, grav, self.cmd * np.array([2.0, 2.0, 0.25]),
                            pos - self.default, vel * 0.05, self.last_a,
                            [math.sin(2 * math.pi * self.phase), math.cos(2 * math.pi * self.phase)]])
        if noise is not None:
            o = o + noise
        return o.astype(np.float32)

    def set_strength(self, k):
        """Scale the motors' torque (domain randomisation)."""
        self.tau_max = np.full(12, M.TAU * k)
        self.push_max = np.full(12, np.inf)
        self.push_max[2::3] = M.A_PUSH * k

    # --- control -------------------------------------------------------------------------------------
    def targets(self, a):
        """Action (12,) in [-1, 1] -> output targets inside the software limits."""
        t = (self.default + a * ACT_SCALE).reshape(4, 3)
        q = np.stack([t[:, 0], t[:, 1], t[:, 2] - t[:, 1]], 1)        # outputs -> (phi, qh, qk)
        q = np.array([G.Leg.clamp(r) for r in q])
        return np.stack([q[:, 0], q[:, 1], q[:, 1] + q[:, 2]], 1).ravel()

    def apply(self, target):
        pos, vel = self.outputs()
        tau = self.kp * (target - pos) - self.kd * vel
        lim = self.tau_max * np.clip(2.0 * (1.0 - np.abs(vel) / M.W0), 0.0, 1.0)
        hi = np.minimum(lim, self.push_max)                            # output A pushing the coupler
        tau = np.minimum(np.maximum(tau, -lim), hi)
        self.d.ctrl[self.act] = tau
        return tau

    def step(self, action, period=None):
        """One policy step: DECIM physics steps. Returns the clipped action, mean |torque| and the largest
        joint-limit force (N*m) seen: the limits are software limits on the robot, a policy must not lean on them."""
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        target = self.targets(a)
        tau_sum = np.zeros(12)
        lim_f = 0.0
        d = self.d
        for _ in range(DECIM):
            tau_sum += np.abs(self.apply(target))
            mujoco.mj_step(self.m, d)
            nefc = d.nefc
            if nefc:
                lm = d.efc_type[:nefc] == LIMIT
                if lm.any():
                    lim_f = max(lim_f, float(np.abs(d.efc_force[:nefc][lm]).max()))
        if period:
            self.phase = (self.phase + DT * DECIM / period) % 1.0
        return a, tau_sum / DECIM, lim_f

    def reset(self, yaw=0.0, dz=0.005):
        """Standing at the origin, joints at zero (the 4-bar and the rotors' equalities hold there), facing yaw."""
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[2] = G.FOOT_R + G.H_STAND - self.legs[0].hip_z + dz
        # yaw about +Z (the robot faces -X at yaw 0)
        d.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        mujoco.mj_forward(m, d)
        self.last_a = np.zeros(12)
        self.phase = 0.0
