"""Gymnasium environment for the v4 quadruped (quad.py): WalkEnv, track a commanded (forward, sideways, yaw rate).

20 s episodes, the command resampled every 4 s (10 % of them zero: stand still). A trot clock (period T_GAIT,
diagonal pairs LR+RF and LF+RR as in gait.py) is in the observation, and feet that follow it are rewarded;
with no command the clock asks for all four feet down. Otherwise legged_gym-style terms: velocity tracking,
level and steady body, height, torques, action rate, foot slip, no load on a joint limit, and -10 for falling
(tipped past 60 deg, torso below 0.22 m, or body or pod on the floor).

Domain randomisation per episode: link masses x0.9-1.1 plus 0-1.5 kg on the torso, floor friction 0.5-1.25,
motor strength x0.85-1.05, PD gains x0.9-1.1, observation noise, and a shove (0.6 m/s) now and then.
"""
import math

import gymnasium as gym
import mujoco
import numpy as np

import quad as Q

T_GAIT, OFF = Q.T_GAIT, Q.OFF
CMD_RANGE = np.array([[-0.4, 1.0], [-0.35, 0.35], [-1.0, 1.0]])
OBS_NOISE = np.concatenate([np.full(3, 0.1), np.full(3, 0.05), np.full(3, 0.03), np.zeros(3),
                            np.full(12, 0.01), np.full(12, 0.05), np.zeros(12), np.zeros(2)])


class WalkEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed=0, episode_s=20.0, randomise=True):
        self.rng = np.random.default_rng(seed)
        self.q = Q.Quad()
        self.randomise = randomise
        self.T = int(episode_s / (Q.DT * Q.DECIM))
        m = self.q.m
        self.base_mass = m.body_mass.copy()
        self.base_fric = m.geom_friction.copy()
        self.action_space = gym.spaces.Box(-1.0, 1.0, (12,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (Q.N_OBS,), np.float32)

    def resample(self):
        lo, hi = CMD_RANGE[:, 0], CMD_RANGE[:, 1]
        self.q.cmd = self.rng.uniform(lo, hi)
        if self.rng.random() < 0.1:
            self.q.cmd[:] = 0.0
        self.q.cmd[np.abs(self.q.cmd) < 0.05] = 0.0

    def noise(self):
        return self.rng.normal(0.0, 1.0, Q.N_OBS) * OBS_NOISE if self.randomise else None

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        q, m = self.q, self.q.m
        if self.randomise:
            m.body_mass[:] = self.base_mass * self.rng.uniform(0.9, 1.1, len(self.base_mass))
            m.body_mass[q.torso] += self.rng.uniform(0.0, 1.5)
            m.geom_friction[:, 0] = self.base_fric[:, 0] * self.rng.uniform(0.5, 1.25)
            q.set_strength(self.rng.uniform(0.85, 1.05))
            q.kp = Q.KP * self.rng.uniform(0.9, 1.1, 12)
            q.kd = Q.KD * self.rng.uniform(0.9, 1.1, 12)
        q.reset(yaw=self.rng.uniform(-math.pi, math.pi))
        for _ in range(25):                                         # settle onto the standing pose (0.5 s)
            q.step(np.zeros(12))
        q.phase = self.rng.random()
        self.t = 0
        self.air = np.zeros(4)
        self.resample()
        return q.obs(self.noise()), {}

    def step(self, action):
        q = self.q
        a, tau, lim_f = q.step(action, T_GAIT)
        self.t += 1
        if self.t % 200 == 0:
            self.resample()
        if self.randomise and self.rng.random() < 0.005:               # a shove now and then
            q.d.qvel[0:2] += self.rng.uniform(-0.6, 0.6, 2)
        R, v, w, grav = q.base()
        contact, bad = q.foot_contacts()
        cmd = q.cmd
        moving = np.linalg.norm(cmd) > 0.0
        fwd_v = -v[0]                                                   # forward is -X
        e_lin = (fwd_v - cmd[0]) ** 2 + (v[1] - cmd[1]) ** 2
        e_yaw = (w[2] - cmd[2]) ** 2
        # trot schedule: stance for the first half of each foot's cycle; all down when standing
        want = ((q.phase + OFF) % 1.0 < 0.5) if moving else np.ones(4, bool)
        r_gait = float(np.mean(contact == want))
        first = contact & (self.air > 0.0)                              # touch-downs this step
        r_air = float(np.sum((self.air - 0.5 * T_GAIT) * first)) if moving else 0.0
        self.air = np.where(contact, 0.0, self.air + Q.DT * Q.DECIM)
        fv = q.foot_vel()
        slip = float(np.sum(contact * np.sum(fv[:, :2] ** 2, 1)))
        z = q.d.qpos[2]
        r = (1.5 * math.exp(-e_lin / 0.25) + 0.75 * math.exp(-e_yaw / 0.25)
             + 0.5 * r_gait + 1.0 * r_air
             - 2.0 * v[2] ** 2 - 0.05 * (w[0] ** 2 + w[1] ** 2)
             - 5.0 * (grav[0] ** 2 + grav[1] ** 2)
             - 20.0 * (z - q.z_stand) ** 2
             - 2e-4 * np.sum(tau ** 2) - 0.01 * np.sum((a - q.last_a) ** 2)
             - 0.2 * slip - 0.05 * lim_f
             - (0.0 if moving else 0.02 * np.sum((q.outputs()[0] - q.default) ** 2) * 10.0))
        q.last_a = a
        fallen = bad or grav[2] > -0.5 or z < 0.22
        if fallen:
            r -= 10.0
        info = {"e_lin": e_lin, "e_yaw": e_yaw}
        return q.obs(self.noise()), float(r) * 0.1, fallen, self.t >= self.T, info
