"""Gymnasium environments for the v3 quadruped (quad_model.py): WalkEnv (track a commanded velocity) and
JumpEnv (one standing long jump, then land and stand).

Control at 50 Hz: the policy gives a target per OUTPUT (ab/ad, output B = thigh, output A = shin's absolute
angle), offsets from the standing pose; a PD loop at 500 Hz turns them into torques, clipped to each
output's torque-speed limit: full torque (the current limit) to half the no-load speed, then falling linearly
to zero at W0 (back-EMF at the supply voltage; v3 first cut held full torque to 0.85 W0, too optimistic for
a jump). The robot's forward is -X.
"""
import math

import gymnasium as gym
import mujoco
import numpy as np

import quad_model as qm

DT, DECIM = 0.002, 10            # physics step, physics steps per policy step (50 Hz)
KP, KD = 45.0, 1.2               # N*m/rad, N*m*s/rad per output
ACT_SCALE = np.array([0.35, 0.8, 0.8] * 4)     # rad: ab/ad, output B, output A
A_PUSH = 25.0                    # N*m: output A's limit in the direction that pushes the coupler (+ here; standing
                                 # holds it at about -5, the coupler pulling). v9.1's coupler buckles at ~1020 N:
                                 # 25 N*m on the 55 mm crank = 455 N, SF 2.2 (full 31.5: SF 1.8)
FWD = np.array([-1.0, 0.0, 0.0])


class QuadBase(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed=0, masses=None, episode_s=20.0):
        self.rng = np.random.default_rng(seed)
        self.masses = masses
        self.model = mujoco.MjModel.from_xml_string(qm.mjcf(masses))
        self.data = mujoco.MjData(self.model)
        self.T = int(episode_s / (DT * DECIM))
        m = self.model
        self.j = {n: [m.joint(f"{p}_{n}").qposadr[0] for p in qm.PODS] for n in ("abad", "hip", "knee")}
        self.jv = {n: [m.joint(f"{p}_{n}").dofadr[0] for p in qm.PODS] for n in ("abad", "hip", "knee")}
        self.act = np.array([[m.actuator(f"{p}_m{c}").id for c in "CBA"] for p in qm.PODS])
        self.feet = [m.geom(f"{p}_foot").id for p in qm.PODS]
        self.floor = m.geom("floor").id
        self.body_geoms = {m.geom("torso_box").id} | {m.geom(f"{p}_podbox").id for p in qm.PODS}
        self.base_mass = m.body_mass.copy()
        self.base_fric = m.geom_friction.copy()
        self.action_space = gym.spaces.Box(-1.0, 1.0, (12,), np.float32)

    # --- outputs (ab/ad, B, A) per pod -----------------------------------------------------------------
    def outputs(self):
        q, v = self.data.qpos, self.data.qvel
        ab = q[self.j["abad"]]
        b = q[self.j["hip"]]
        a = b + q[self.j["knee"]]
        dab, db = v[self.jv["abad"]], v[self.jv["hip"]]
        da = db + v[self.jv["knee"]]
        return np.stack([ab, b, a], 1).ravel(), np.stack([dab, db, da], 1).ravel()

    def apply(self, target):
        pos, vel = self.outputs()
        tau = KP * (target - pos) - KD * vel
        lim = qm.TAU * np.clip(2.0 * (1.0 - np.abs(vel) / qm.W0), 0.0, 1.0)   # current limit to W0/2, then the voltage line to 0 at W0
        hi = lim.copy()
        hi[2::3] = np.minimum(hi[2::3], A_PUSH)                     # output A pushing the coupler
        tau = np.clip(tau, -lim, hi)
        self.data.ctrl[self.act.ravel()] = tau
        return tau

    def base(self):
        d = self.data
        quat = d.qpos[3:7]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        R = R.reshape(3, 3)
        v_body = R.T @ d.qvel[0:3]
        w_body = d.qvel[3:6].copy()                               # free joint angular velocity is in the body frame
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        return R, v_body, w_body, grav

    def foot_contacts(self):
        c = np.zeros(4, bool)
        bad = False
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            gs = {con.geom1, con.geom2}
            if self.floor in gs:
                o = (gs - {self.floor}).pop() if len(gs) > 1 else self.floor
                if o in self.feet:
                    c[self.feet.index(o)] = True
                elif o in self.body_geoms:
                    bad = True
        return c, bad

    def reset_robot(self, noise=0.02):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = qm.STAND_Z + 0.005
        self.data.qpos[7:] += self.rng.uniform(-noise, noise, self.model.nq - 7)
        m = self.model                                            # domain randomisation
        m.body_mass[:] = self.base_mass * self.rng.uniform(0.9, 1.1, len(self.base_mass))
        m.geom_friction[:, 0] = self.base_fric[:, 0] * self.rng.uniform(0.6, 1.25)
        mujoco.mj_forward(m, self.data)
        self.last_a = np.zeros(12)
        self.t = 0

    def sim(self, action):
        a = np.clip(action, -1, 1)
        target = a * ACT_SCALE
        self.step_grf = 0.0
        self.step_lim = 0.0
        taus = []
        for _ in range(DECIM):
            taus.append(self.apply(target))
            mujoco.mj_step(self.model, self.data)
            d_ = self.data                                            # joint-limit torque: the knee's +-60 and the outputs'
            for k_ in range(d_.nefc):                                 # +-90 are SOFTWARE limits on the robot (past them:
                if d_.efc_type[k_] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT or \
                        d_.efc_type[k_] == mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON:   # dead centre, the rope's anchors),
                    self.step_lim = max(self.step_lim, abs(d_.efc_force[k_]))       # so a policy must not lean on them
            if getattr(self, "grf_on", False):                        # jump: the largest foot force this control step
                f6 = np.zeros(6)
                for i in range(self.data.ncon):
                    con = self.data.contact[i]
                    if con.geom1 in self.feet or con.geom2 in self.feet:
                        mujoco.mj_contactForce(self.model, self.data, i, f6)
                        self.step_grf = max(self.step_grf, abs(f6[0]))
            if getattr(self, "track", False):                         # evaluation only: peak foot forces, saturation
                f6 = np.zeros(6)
                for i in range(self.data.ncon):
                    con = self.data.contact[i]
                    if con.geom1 in self.feet or con.geom2 in self.feet:
                        mujoco.mj_contactForce(self.model, self.data, i, f6)
                        j = self.feet.index(con.geom1 if con.geom1 in self.feet else con.geom2)
                        pk = self.peak_land if getattr(self, "took_off", False) else self.peak_grf    # landing vs push-off
                        pk[j] = max(pk[j], abs(f6[0]))
                        if hasattr(self, "contact_log"):                  # world force on the foot, for the leg's loads
                            fw = con.frame.reshape(3, 3).T @ f6[:3]
                            self.contact_log.append((j, fw if con.geom1 == self.floor else -fw, self.data.qpos.copy(),
                                                     bool(getattr(self, "took_off", False))))
                self.peak_tau = np.maximum(self.peak_tau, np.abs(taus[-1]))
        return a, np.mean(np.abs(np.array(taus)), 0)


class WalkEnv(QuadBase):
    """Track (forward speed, sideways speed, yaw rate) commands; 20 s episodes, commands resampled every 5 s."""

    def __init__(self, seed=0, masses=None):
        super().__init__(seed, masses, 20.0)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (48,), np.float32)

    def resample(self):
        self.cmd = np.array([self.rng.uniform(-0.2, 1.0), self.rng.uniform(-0.3, 0.3), self.rng.uniform(-0.8, 0.8)])
        if self.rng.random() < 0.1:
            self.cmd[:] = 0.0

    def obs(self):
        R, v, w, grav = self.base()
        pos, vel = self.outputs()
        return np.concatenate([v * 2.0, w * 0.25, grav, self.cmd, pos, vel * 0.05, self.last_a]).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.reset_robot()
        self.resample()
        self.air = np.zeros(4)
        return self.obs(), {}

    def step(self, action):
        a, tau = self.sim(action)
        self.t += 1
        if self.t % 250 == 0:
            self.resample()
        if self.rng.random() < 0.004:                               # a shove now and then
            self.data.qvel[0:2] += self.rng.uniform(-0.5, 0.5, 2)
        R, v, w, grav = self.base()
        contact, bad = self.foot_contacts()
        fwd_v = -v[0]                                               # the robot's forward is -X
        e_lin = (fwd_v - self.cmd[0]) ** 2 + (v[1] - self.cmd[1]) ** 2
        e_yaw = (w[2] - self.cmd[2]) ** 2
        first = contact & (self.air > 0.0)
        r_air = np.sum((self.air - 0.25) * first) if np.linalg.norm(self.cmd) > 0.1 else 0.0
        self.air = np.where(contact, 0.0, self.air + DT * DECIM)
        z = self.data.qpos[2]
        r = (1.5 * math.exp(-e_lin / 0.25) + 0.75 * math.exp(-e_yaw / 0.25)
             - 2.0 * v[2] ** 2 - 0.05 * (w[0] ** 2 + w[1] ** 2)
             - 1.0 * (grav[0] ** 2 + grav[1] ** 2)
             - 5e-4 * np.sum(tau ** 2) - 0.01 * np.sum((a - self.last_a) ** 2)
             + 1.0 * r_air - 2.0 * max(0.0, 0.30 - z)
             - 0.05 * self.step_lim)                                    # no standing on a joint limit (N*m)
        self.last_a = a
        fallen = bad or grav[2] > -0.5 or z < 0.18
        if fallen:
            r -= 10.0
        return self.obs(), float(r) * DT * DECIM * 50.0 / 50.0, fallen, self.t >= self.T, {"cmd": self.cmd.copy()}


class JumpEnv(QuadBase):
    """One standing long jump toward -X, then land softly and stand: 3 s episodes. Paid for the distance
    covered IN THE AIR (take-off to first touchdown), once, at that touchdown; before take-off only upward
    speed is rewarded, and not leaving the ground by 1.5 s costs 5. Foot forces over GRF_SOFT and any load on
    a joint limit are penalised; so is bounding on after landing. History of what each earlier reward taught:
    distance step by step -> a 0.19 m hop; 0.4 a step for standing after landing -> 0.10 m; distance paid at
    an all-four-feet landing -> bounding for 3 s; forward speed before take-off -> trotting, never jumping."""

    GRF_SOFT = 400.0                                                 # N per foot
    FLIGHT = 0.08                                                    # s with all four feet off: a real take-off

    def __init__(self, seed=0, masses=None, land_practice=None):
        super().__init__(seed, masses, 3.0)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (48,), np.float32)
        self.grf_on = True
        import os
        self.land_practice = bool(int(os.environ.get("JUMP_LAND_PRACTICE", "0"))) if land_practice is None else land_practice
        self.target = 0.05                                              # distance curriculum (train.py raises it)

    def set_target(self, d):
        self.target = d

    def obs(self):
        R, v, w, grav = self.base()
        pos, vel = self.outputs()
        phase = self.t / self.T
        return np.concatenate([v * 2.0, w * 0.25, grav, [self.data.qpos[2] - qm.STAND_Z, phase],
                               pos, vel * 0.05, self.last_a, [float(self.landed)]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.reset_robot(0.01)
        self.took_off = False
        self.landed = False
        self.air = 0.0
        self.flight = 0.0
        self.x_off = None
        self.practice = bool(self.land_practice and self.rng.random() < 0.3)
        if self.practice:                                               # some episodes start mid-jump: landing practice
            d = self.data
            d.qpos[2] = qm.STAND_Z + self.rng.uniform(0.05, 0.20)
            d.qvel[0] = -self.rng.uniform(0.8, 2.2)                     # forward is -X
            d.qvel[2] = self.rng.uniform(-1.5, 0.3)
            d.qvel[4] = self.rng.uniform(-1.0, 1.0)                     # some pitch rate
            mujoco.mj_forward(self.model, d)
            self.took_off, self.x_off, self.air, self.x_air = True, d.qpos[0], self.FLIGHT, d.qpos[0]
        self.air_after = 0.0
        self.jump = 0.0
        self.bottomed = False
        return self.obs(), {}

    def step(self, action):
        a, tau = self.sim(action)
        self.t += 1
        R, v, w, grav = self.base()
        contact, bad = self.foot_contacts()
        airborne = not contact.any()
        vw = self.data.qvel[0:3]                                     # world linear velocity
        x = self.data.qpos[0]
        r = 0.0
        if not self.landed:
            if airborne:
                if self.air == 0.0:
                    self.x_air = x                                       # where this airborne spell began
                self.air += DT * DECIM
                if self.air >= self.FLIGHT and not self.took_off:
                    self.took_off, self.x_off = True, self.x_air
                r += 1.0 * max(0.0, self.data.qpos[2] - qm.STAND_Z) if self.took_off else 0.0   # height while flying
            else:
                if self.took_off:                                        # first touchdown after a real flight
                    self.landed, self.flight = True, self.air
                    self.jump = max(0.0, -(x - self.x_off))
                    r += 40.0 * self.jump                                # the distance flown, once
                self.air = 0.0
            if not self.took_off:
                r += 0.3 * max(0.0, vw[2])                              # push upward
                if self.t == int(1.5 / (DT * DECIM)):
                    r -= 5.0                                            # jump in the first 1.5 s
        upright = -grav[2]
        if self.landed:
            r += 0.3 * upright - 0.05 * np.linalg.norm(vw)              # on its feet after landing (0.1: it would jump 0.74 m
                                                                        # and tip over: standing was worth less than the jump)
            self.air_after = self.air_after + DT * DECIM if airborne else 0.0
            if self.air_after > 0.06:
                r -= 0.5                                                # no bounding on after the jump
        r -= 0.02 * max(0.0, self.step_grf - self.GRF_SOFT)             # soft landings: N over the design load
        r -= 0.2 * self.step_lim                                        # no landing on a joint limit (N*m)
        r -= 0.3 * (grav[0] ** 2 + grav[1] ** 2) + 2e-4 * np.sum(tau ** 2) + 0.005 * np.sum((a - self.last_a) ** 2)
        self.last_a = a
        if bad:
            r -= 3.0                                                    # body or pod on the ground: heavy, but not the end
            self.bottomed = True                                        # (so it can learn what follows a rough landing)
        fallen = upright < 0.5                                          # tipped over: the end
        if fallen:
            r -= 10.0
        done = self.t >= self.T
        if done and self.landed and not fallen and not self.bottomed:
            r += 10.0
            if self.jump >= self.target:                                # curriculum: a clean landing at the target
                r += 30.0
        info = {"distance": self.jump, "flight": self.flight, "landed": self.landed, "bottomed": self.bottomed,
                "practice": self.practice}
        return self.obs(), float(r), fallen, done, info
