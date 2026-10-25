"""The walking task on the GPU: thousands of v4 quadrupeds stepped together by MuJoCo Warp (through MJX).

The same robot, task and interface as quad.py / envs.py (the observation is built identically, so a policy trained
here runs in quad.py's full model unchanged), with two differences in the physics model:
  - the visual meshes are dropped (they never collide; the GPU need not carry 178 meshes);
  - the motor rotors are welded to their links: their spin is already in the drives' armature (model.py), and the
    12 joint equalities (-9x mimics) cost solver time for no change in the dynamics but the rotors' gyroscopics.

Episodes are TASK CONFIGURATIONS from a fixed bank (train_gpu.py samples them; see the SGS sampler there): two
commands (4 s + 4 s of 8 s), floor friction, added torso mass, motor strength, PD gain scales, and one shove.
An episode succeeds if the robot neither falls nor misses its commands: rms velocity error under 0.2 m/s and
yaw-rate error under 0.35 rad/s, measured after the first 0.6 s of each command.
"""
import math
import sys
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

import quad as Q

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mujoco"))
import gait as G                                                     # noqa: E402
import model as M                                                    # noqa: E402

EP_STEPS = 400                                   # 8 s at 50 Hz
SWITCH = 200                                     # the second command from here
SETTLE, OK_LIN, OK_YAW = Q.SETTLE, Q.OK_LIN, Q.OK_YAW
NACON, NJMAX = 24, 160                           # contacts and constraint rows per world (Warp's buffers)
OBS_NOISE = np.asarray(np.concatenate([np.full(3, 0.1), np.full(3, 0.05), np.full(3, 0.03), np.zeros(3),
                                       np.full(12, 0.01), np.full(12, 0.05), np.zeros(14)]), np.float32)
N_PRIV = 10                                      # the critic's extra inputs (privileged: the configuration, contacts)


def physics_model():
    spec = M.build_spec()
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
    return spec.compile()


def sample_bank(n, seed=0, kind="base"):
    """n task configurations from quad.BANKS[kind], as a dict of arrays (n, ...)."""
    r = np.random.default_rng(seed)
    b = Q.BANKS[kind]
    lo, hi = np.array(b["cmd"]).T

    def cmds():
        c = r.uniform(lo, hi, (n, 3))
        c[r.random(n) < 0.12] = 0.0                                    # stand still
        one = r.random(n) < 0.3                                        # one axis only (straight walk, side-step, spin)
        keep = r.integers(0, 3, n)
        c[one] *= np.eye(3)[keep[one]]
        c[np.abs(c) < 0.05] = 0.0
        return c
    f = lambda ab: r.uniform(ab[0], ab[1], n)                          # noqa: E731
    push_dir = f([-math.pi, math.pi])
    push = f(b["push"]) * (r.random(n) < b["p_push"])
    bank = dict(cmd_a=cmds(), cmd_b=cmds(), fric=f(b["fric"]), mass_add=f(b["mass"]), strength=f(b["strength"]),
                kp_s=f(b["kp"]), kd_s=f(b["kd"]), push_v=np.stack([np.cos(push_dir), np.sin(push_dir)], 1) * push[:, None],
                push_t=r.integers(60, EP_STEPS - 40, n), yaw0=f([-math.pi, math.pi]))
    return {k: jp.asarray(v, jp.int32 if k == "push_t" else jp.float32) for k, v in bank.items()}


class Walk:
    def __init__(self, n_env):
        self.n = n_env
        m = self.m = physics_model()
        self.mx = mjx.put_model(m, impl="warp")
        legs = [G.Leg(m, p) for p in M.PODS]
        self.qa = jp.array(np.array([lg.ja for lg in legs]).ravel())
        self.va = jp.array(np.array([lg.jv for lg in legs]).ravel())
        self.act = jp.array(np.array([lg.act for lg in legs]).ravel())
        self.foot_site = jp.array([m.site(f"foot_{p}").id for p in M.PODS])
        self.torso = m.body("torso").id
        q = np.array([lg.ik(lg.nominal(Q.H_NOM)) for lg in legs])
        self.default = jp.array(np.stack([q[:, 0], q[:, 1], q[:, 1] + q[:, 2]], 1).ravel(), jp.float32)
        self.z_stand = float(G.FOOT_R + Q.H_NOM - legs[0].hip_z)
        self.z0 = float(G.FOOT_R + G.H_STAND - legs[0].hip_z + 0.005)    # reset: joints at zero
        self.fric0 = jp.array(m.geom_friction)
        self.mass0 = jp.array(m.body_mass)
        self.in_axes = jax.tree_util.tree_map(lambda x: None, self.mx).tree_replace(
            {"geom_friction": 0, "body_mass": 0})
        self.kp, self.kd, self.scale = jp.array(Q.KP), jp.array(Q.KD), jp.array(Q.ACT_SCALE)
        self.isA = jp.arange(12) % 3 == 2

    # --- data -----------------------------------------------------------------------------------------
    def make_data(self):
        def mk(_):
            return mjx.make_data(self.m, impl="warp", naconmax=NACON * self.n, njmax=NJMAX)
        return jax.vmap(mk)(jp.arange(self.n))

    def reset_slot(self, bank, i, key):
        """Fresh state for one env on configuration i: (qpos, qvel, aux, friction, mass)."""
        c = {k: v[i] for k, v in bank.items()}
        qpos = jp.zeros(self.m.nq).at[2].set(self.z0)
        y = c["yaw0"]
        qpos = qpos.at[3:7].set(jp.array([jp.cos(y / 2), 0.0, 0.0, jp.sin(y / 2)]))
        aux = dict(cmd=c["cmd_a"], phase=jax.random.uniform(key), last_a=jp.zeros(12), t=jp.int32(0), air=jp.zeros(4),
                   e_lin=jp.float32(0), e_yaw=jp.float32(0), n_err=jp.float32(0), cfg=jp.int32(i),
                   kp=self.kp * c["kp_s"], kd=self.kd * c["kd_s"], strength=c["strength"],
                   priv_c=jp.array([c["fric"], c["mass_add"], c["strength"], c["kp_s"], c["kd_s"]]))
        fric = self.fric0.at[:, 0].multiply(c["fric"])
        mass = self.mass0.at[self.torso].add(c["mass_add"])
        return qpos, jp.zeros(self.m.nv), aux, fric, mass

    # --- one env (vmapped) ------------------------------------------------------------------------------
    def outputs(self, d):
        pos, vel = d.qpos[self.qa], d.qvel[self.va]
        return pos.at[2::3].add(pos[1::3]), vel.at[2::3].add(vel[1::3])

    def targets(self, a):
        """quad.Quad.targets: action -> output targets inside gait.Leg.clamp's limits."""
        t = (self.default + a * self.scale).reshape(4, 3)
        ab = jp.clip(t[:, 0], -G.ABAD_LIM, G.ABAD_LIM)
        b = jp.clip(t[:, 1], -G.OUT_LIM, G.OUT_LIM)
        a_ = jp.clip(t[:, 2], -G.OUT_LIM, G.OUT_LIM)
        a_ = b + jp.clip(a_ - b, -G.KNEE_SOFT, G.KNEE_SOFT)
        return jp.stack([ab, b, a_], 1).ravel()

    def base(self, d):
        """From qpos / qvel only (right straight after a reset, before any forward pass)."""
        q = d.qpos[3:7] / jp.linalg.norm(d.qpos[3:7])
        w_, x, y, z = q
        R = jp.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w_ * z), 2 * (x * z + w_ * y)],
                      [2 * (x * y + w_ * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w_ * x)],
                      [2 * (x * z - w_ * y), 2 * (y * z + w_ * x), 1 - 2 * (x * x + y * y)]])
        return R, R.T @ d.qvel[0:3], d.qvel[3:6], -R[2, :]

    def obs(self, d, aux, key):
        R, v, w, grav = self.base(d)
        pos, vel = self.outputs(d)
        ph = 2 * jp.pi * aux["phase"]
        o = jp.concatenate([v * 2.0, w * 0.25, grav, aux["cmd"] * jp.array([2.0, 2.0, 0.25]), pos - self.default,
                            vel * 0.05, aux["last_a"], jp.array([jp.sin(ph), jp.cos(ph)])])
        return o + jax.random.normal(key, o.shape) * OBS_NOISE

    def step(self, mx, d, aux, action, bank, key):
        a = jp.clip(action, -1.0, 1.0)
        target = self.targets(a)
        k_push, k_obs = jax.random.split(key)
        c_push_t, c_push_v = bank["push_t"][aux["cfg"]], bank["push_v"][aux["cfg"]]
        d = d.replace(qvel=jp.where(aux["t"] == c_push_t, d.qvel.at[0:2].add(c_push_v), d.qvel))
        foot0 = d.site_xpos[self.foot_site]
        tau_max = M.TAU * aux["strength"]
        push_max = jp.where(self.isA, M.A_PUSH * aux["strength"], jp.inf)

        def sub(d, _):
            pos, vel = self.outputs(d)
            tau = aux["kp"] * (target - pos) - aux["kd"] * vel
            lim = tau_max * jp.clip(2.0 * (1.0 - jp.abs(vel) / M.W0), 0.0, 1.0)
            tau = jp.minimum(jp.maximum(tau, -lim), jp.minimum(lim, push_max))
            d = mjx.step(mx, d.replace(ctrl=d.ctrl.at[self.act].set(tau)))
            return d, jp.abs(tau)
        d, taus = jax.lax.scan(sub, d, None, length=Q.DECIM)
        tau = taus.mean(0)

        t = aux["t"] + 1
        cmd = jp.where(t >= SWITCH, bank["cmd_b"][aux["cfg"]], aux["cmd"])
        phase = (aux["phase"] + Q.DT * Q.DECIM / Q.T_GAIT) % 1.0
        R, v, w, grav = self.base(d)
        foot = d.site_xpos[self.foot_site]
        contact = foot[:, 2] < G.FOOT_R + 0.004
        fv = (foot - foot0) / (Q.DT * Q.DECIM)
        moving = jp.linalg.norm(cmd) > 0.0
        e_lin = (-v[0] - cmd[0]) ** 2 + (v[1] - cmd[1]) ** 2
        e_yaw = (w[2] - cmd[2]) ** 2
        want = jp.where(moving, (phase + Q.OFF) % 1.0 < 0.5, True)
        r_gait = jp.mean(contact == want)
        first = contact & (aux["air"] > 0.0)
        r_air = jp.where(moving, jp.sum((aux["air"] - 0.5 * Q.T_GAIT) * first), 0.0)
        air = jp.where(contact, 0.0, aux["air"] + Q.DT * Q.DECIM)
        slip = jp.sum(contact * jp.sum(fv[:, :2] ** 2, 1)) * (t > 1)        # foot0 is stale on a reset's first step
        z = d.qpos[2]
        pos, _ = self.outputs(d)
        r = (1.5 * jp.exp(-e_lin / 0.25) + 0.75 * jp.exp(-e_yaw / 0.25) + 0.5 * r_gait + 1.0 * r_air
             - 2.0 * v[2] ** 2 - 0.05 * (w[0] ** 2 + w[1] ** 2) - 5.0 * (grav[0] ** 2 + grav[1] ** 2)
             - 20.0 * (z - self.z_stand) ** 2 - 2e-4 * jp.sum(tau ** 2) - 0.01 * jp.sum((a - aux["last_a"]) ** 2)
             - 0.2 * slip - jp.where(moving, 0.0, 0.2 * jp.sum((pos - self.default) ** 2)))
        bad = ~jp.all(jp.isfinite(d.qpos)) | ~jp.all(jp.isfinite(d.qvel))
        fallen = bad | (grav[2] > -0.5) | (z < 0.22)
        scored = (t % SWITCH) >= SETTLE
        e_lin_s = aux["e_lin"] + jp.where(scored, e_lin, 0.0)
        e_yaw_s = aux["e_yaw"] + jp.where(scored, e_yaw, 0.0)
        n_err = aux["n_err"] + scored
        trunc = (t >= EP_STEPS) & ~fallen
        rms_lin, rms_yaw = jp.sqrt(e_lin_s / jp.maximum(n_err, 1)), jp.sqrt(e_yaw_s / jp.maximum(n_err, 1))
        success = trunc & (rms_lin < OK_LIN) & (rms_yaw < OK_YAW)
        r = r - 10.0 * fallen + 10.0 * success
        aux = dict(aux, cmd=cmd, phase=phase, last_a=a, t=t, air=air, e_lin=e_lin_s, e_yaw=e_yaw_s, n_err=n_err)
        o = self.obs(d, aux, k_obs)
        priv = jp.concatenate([aux["priv_c"], jp.array([z - self.z_stand]), contact.astype(jp.float32)])
        info = dict(success=success, fallen=fallen, rms_lin=rms_lin, rms_yaw=rms_yaw, e_lin=e_lin)
        r = jp.where(bad, -10.0, r)
        o = jp.where(bad, 0.0, o)
        return d, aux, o, priv, r * 0.1, fallen | trunc, trunc, info
