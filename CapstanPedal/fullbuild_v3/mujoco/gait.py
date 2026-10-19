"""Model-based locomotion controller for the URDF quadruped (model.py): no learning.

Walking is a trot (diagonal pairs) with
  - foot targets in a gravity-aligned heading frame, so the body stays level whatever the legs do;
  - stance feet fixed on the ground (they slide back under the body at the commanded velocity and yaw rate);
  - swing feet placed by the Raibert heuristic (half a stance of travel ahead, plus a correction for
    velocity error), lifted on a smooth arc;
  - full 3-D leg inverse kinematics (ab/ad, then the 2-link leg in its plane), joint PD on the drive
    OUTPUTS (B = hip, A = hip + knee, the 4-bar's crank) and feed-forward J^T F: each stance leg carries its
    share of the weight plus a roll/pitch correction;
  - the same motor model as the RL work: 31.5 N*m, the torque-speed line from half the no-load speed, output
    A limited to 25 N*m in the direction that pushes the coupler.
It stands still (four feet down) when there is no command.

Jumping is the scripted jump of rl/scripted_jump.py (crouch, force-controlled push along THETA, swing to a
landing pose set in the world frame, soft landing), with its tuned parameters (rl/scripted_jump.json).
"""
import json
import math
from pathlib import Path

import mujoco
import numpy as np

import model as M

HERE = Path(__file__).resolve().parent
FR = json.load(open(HERE.parent / "urdf" / "frames.json"))
L = 0.200                                    # thigh, shin (hip to knee, knee to foot centre)
Q0T, Q0S = math.radians(-60.0), math.radians(-120.0)   # thigh and shin angles at joint zero (from +X toward +Z)
FOOT_R = 0.023
KNEE_SOFT = math.radians(52.0)
H_STAND = 2 * L * math.sin(math.radians(60.0))        # hip to foot centre at the rest pose (0.346)
H_RANGE = (0.26, 0.385)                               # hip height above foot centre the controller accepts
DIAG = {"LF": 0.0, "RR": 0.0, "RF": 0.5, "LR": 0.5}   # trot phase offsets


class Leg:
    def __init__(self, m, p):
        self.p = p
        f = FR[p]
        self.ab_y, self.ab_z = f["ab"][1], f["ab"][2]
        self.hip_x, self.hip_z = f["hip"][0], f["hip"][2]
        self.d0 = f["leg_y"] - self.ab_y                 # leg plane's offset from the ab/ad axis
        self.leg_y = f["leg_y"]
        self.ja = [m.joint(f"{p}_{n}").qposadr[0] for n in ("abad", "hip", "knee")]
        self.jv = [m.joint(f"{p}_{n}").dofadr[0] for n in ("abad", "hip", "knee")]
        self.act = [m.actuator(f"{p}_m{c}").id for c in "CBA"]
        self.foot_geom = m.geom(f"foot_{p}").id

    def fk(self, phi, qh, qk):
        """Foot centre (body frame) and its Jacobian d(x, y, z)/d(phi, qh, qk)."""
        th, sh = Q0T - qh, Q0S - qh - qk
        ct, st, cs, ss = math.cos(th), math.sin(th), math.cos(sh), math.sin(sh)
        x = self.hip_x + L * (ct + cs)
        zl = (self.hip_z - self.ab_z) + L * (st + ss)
        yl = self.d0
        cp, sp = math.cos(phi), math.sin(phi)
        y = yl * cp - zl * sp
        z = yl * sp + zl * cp
        dx = (L * (st + ss), L * ss)
        dzl = (-L * (ct + cs), -L * cs)
        J = np.array([[0.0, dx[0], dx[1]],
                      [-z, -sp * dzl[0], -sp * dzl[1]],
                      [y, cp * dzl[0], cp * dzl[1]]])
        return np.array([x, y + self.ab_y, z + self.ab_z]), J

    def ik(self, p):
        """Foot centre p (body frame) -> (phi, qh, qk), on the rest pose's knee branch, inside the soft knee range."""
        dy, dz = p[1] - self.ab_y, p[2] - self.ab_z
        r2 = dy * dy + dz * dz
        zl = -math.sqrt(max(r2 - self.d0 * self.d0, 1e-6))
        phi = math.atan2(dz, dy) - math.atan2(zl, self.d0)
        x, z = p[0] - self.hip_x, zl - (self.hip_z - self.ab_z)
        d = min(math.hypot(x, z), 2 * L * 0.999)
        ang = math.atan2(z, x)
        al = math.acos(d / (2 * L))
        th, sh = ang + al, ang - al
        qh = Q0T - th
        qk = (Q0S - sh) - qh
        qk = min(max(qk, -KNEE_SOFT), KNEE_SOFT)
        return phi, qh, qk

    def nominal(self, h, out=0.0):
        """Foot centre under the hip, h below it (body frame, level)."""
        side = 1.0 if self.leg_y > 0 else -1.0
        return np.array([self.hip_x, self.leg_y + side * out, self.hip_z - h])


class Controller:
    T_GAIT = 0.25                    # s per trot cycle (tuned in sim: peak tilt ~5 deg over walk, turn, side-step)
    DUTY = 0.6                       # each foot down 60 % of the cycle: both pairs share the load at the changes
    CLEAR = 0.05                     # m swing height
    KP, KD = (90.0, 70.0, 70.0), (1.6, 3.0, 3.0)        # stance: ab/ad, B, A (N*m/rad, N*m*s/rad)
    KP_SW, KD_SW = (60.0, 45.0, 45.0), (1.2, 1.0, 1.0)  # swing
    K_RAIBERT = 0.03
    K_V = 2.0                        # integral of the velocity error, added to the stance feet's slide
    K_ATT = (1500.0, 150.0)          # roll/pitch PD -> vertical force split (N/rad, N*s/rad)

    def __init__(self, m, d):
        self.m, self.d = m, d
        self.legs = {p: Leg(m, p) for p in M.PODS}
        self.mass = float(m.body_subtreemass[m.body("torso").id])
        self.floor = m.geom("floor").id
        self.cmd = np.zeros(3)                         # forward m/s (robot's forward is -X), left m/s, yaw rad/s
        self.h = H_STAND - 0.01
        self.phase = 0.0
        self.walking = False
        self.idle = 0.0
        self.stance = {p: True for p in M.PODS}
        self.foot = {p: self.legs[p].nominal(self.h) for p in M.PODS}       # heading-frame foot targets (rel. body)
        self.lift = {p: self.foot[p].copy() for p in M.PODS}
        self.jump_state = None
        try:                                           # tuned on this model (jump_params.json), else the RL model's
            self.jp = json.load(open(HERE / "jump_params.json"))
        except FileNotFoundError:
            self.jp = json.load(open(HERE.parent / "rl" / "scripted_jump.json"))
        if not self.jp:
            self.jp = dict(D_CROUCH=0.221, X_C=-0.007, D_EXT=0.37, THETA=59.3, T_CROUCH=0.6, LAND_ANG=-92.3,
                           D_LAND=0.335, KP_LAND=56.8, KD_LAND=2.2, LAND_GIVE=0.21, FRONT_EXTRA=-0.014, KD_PUSH=0.45)

    # --- state ---------------------------------------------------------------------------------------
    def body(self):
        d = self.d
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        R = R.reshape(3, 3)
        yaw = math.atan2(R[1, 0], R[0, 0])
        cy, sy = math.cos(yaw), math.sin(yaw)
        Ryaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Rrp = Ryaw.T @ R                                      # roll/pitch only: body -> heading
        v_head = Ryaw.T @ d.qvel[0:3]
        w_body = d.qvel[3:6]
        roll = math.atan2(Rrp[2, 1], Rrp[2, 2])
        pitch = math.atan2(-Rrp[2, 0], math.hypot(Rrp[2, 1], Rrp[2, 2]))
        return R, Rrp, v_head, w_body, roll, pitch

    def contacts(self):
        c = {p: False for p in M.PODS}
        ids = {self.legs[p].foot_geom: p for p in M.PODS}
        for i in range(self.d.ncon):
            con = self.d.contact[i]
            if con.geom1 == self.floor and con.geom2 in ids:
                c[ids[con.geom2]] = True
            elif con.geom2 == self.floor and con.geom1 in ids:
                c[ids[con.geom1]] = True
        return c

    def reset_pose(self):
        """Put the robot standing at the origin."""
        d = self.d
        mujoco.mj_resetData(self.m, d)
        d.qpos[2] = FOOT_R + self.h - list(self.legs.values())[0].hip_z
        mujoco.mj_forward(self.m, d)
        self.phase, self.walking, self.jump_state = 0.0, False, None
        self.foot = {p: self.legs[p].nominal(self.h) for p in M.PODS}
        self.stance = {p: True for p in M.PODS}

    # --- commands ------------------------------------------------------------------------------------
    def jump(self):
        if self.jump_state is None:
            self.jump_state = dict(phase="crouch", t0=self.d.time, off=0)

    # --- control -------------------------------------------------------------------------------------
    def torques(self, leg, q_t, kp, kd, f_ff=None, vel_lim=True):
        """Output-space PD toward joint targets q_t = (phi, qh, qk), plus J^T f_ff (force the GROUND puts on the foot,
        body frame), clipped to the motor model. Writes ctrl."""
        d = self.d
        phi, qh, qk = (d.qpos[a] for a in leg.ja)
        dphi, dqh, dqk = (d.qvel[a] for a in leg.jv)
        out = (phi, qh, qh + qk)
        dout = (dphi, dqh, dqh + dqk)
        tgt = (q_t[0], q_t[1], q_t[1] + q_t[2])
        tau = [kp[i] * (tgt[i] - out[i]) - kd[i] * dout[i] for i in range(3)]
        if f_ff is not None:
            _, J = leg.fk(phi, qh, qk)
            tj = -(J.T @ f_ff)                              # joint torques that hold the ground's push
            tau[0] += tj[0]
            tau[1] += tj[1] - tj[2]                         # output B carries hip minus knee
            tau[2] += tj[2]                                 # output A (tendon hip + knee) carries the knee
        for i in range(3):
            lim = M.TAU * min(1.0, max(0.0, 2.0 * (1.0 - abs(dout[i]) / M.W0)))
            hi = min(lim, M.A_PUSH) if i == 2 else lim
            tau[i] = min(max(tau[i], -lim), hi)
            d.ctrl[leg.act[i]] = tau[i]
        return tau

    def step(self, dt):
        """One control step (call every physics step)."""
        if self.jump_state is not None:
            return self.step_jump(dt)
        R, Rrp, v, w, roll, pitch = self.body()
        cmd = self.cmd
        moving = np.linalg.norm(cmd) > 0.02 or math.hypot(v[0], v[1]) > 0.15
        self.idle = 0.0 if moving else self.idle + dt
        if moving and not self.walking:
            self.walking, self.phase = True, 0.0
        if self.walking and self.idle > 0.4 and (self.phase + 0.5 - self.DUTY) % 0.5 < 0.02:
            self.walking = False                                    # stop at a pair change
        # heading-frame velocity command: robot forward is -X
        vc = np.array([-cmd[0], cmd[1], 0.0])
        if self.walking:                                            # stance feet slide a little faster when the body lags
            self.v_int = getattr(self, "v_int", np.zeros(3)) + (vc - np.array([v[0], v[1], 0.0])) * dt
            self.v_int = np.clip(self.v_int, -0.15, 0.15)
        else:
            self.v_int = np.zeros(3)
        vc_ground = vc + self.K_V * self.v_int
        wz = cmd[2]
        Ts = self.T_GAIT * self.DUTY
        if self.walking:
            self.phase = (self.phase + dt / self.T_GAIT) % 1.0
        n_st = 0
        for p, leg in self.legs.items():
            ph = (self.phase + DIAG[p]) % 1.0
            st = (not self.walking) or ph < self.DUTY
            if not st and self.stance[p]:
                self.lift[p] = self.foot[p].copy()                  # lift-off
            self.stance[p] = st
            n_st += st
        weight = self.mass * 9.81 / max(n_st, 1)
        # attitude correction: more push under the side that is low
        fz_roll = -self.K_ATT[0] * roll - self.K_ATT[1] * w[0]
        fz_pitch = -self.K_ATT[0] * pitch - self.K_ATT[1] * w[1]
        for p, leg in self.legs.items():
            nom = leg.nominal(self.h, 0.0)
            nom[2] = leg.hip_z - self.h
            if self.stance[p]:
                f = self.foot[p]
                f[0] -= (vc_ground[0] - wz * f[1]) * dt              # the ground slides back under the body
                f[1] -= (vc_ground[1] + wz * f[0]) * dt
                f[2] += (nom[2] - f[2]) * min(1.0, 5.0 * dt)        # height follows R/F
                if not self.walking:                                 # standing: feet settle back under the hips
                    f[0] += (nom[0] - f[0]) * min(1.0, 2.0 * dt)
                    f[1] += (nom[1] - f[1]) * min(1.0, 2.0 * dt)
                sgn_x = 1.0 if leg.hip_x > 0 else -1.0
                sgn_y = 1.0 if leg.leg_y > 0 else -1.0
                fz = weight + sgn_y * fz_roll / 2.0 + sgn_x * (-fz_pitch) / 2.0
                f_ff = Rrp.T @ np.array([0.0, 0.0, max(fz, 0.0)])
                pb = Rrp.T @ f                                       # heading -> body
                q = leg.ik(pb)
                self.torques(leg, q, self.KP, self.KD, f_ff)
            else:
                ph = (self.phase + DIAG[p]) % 1.0
                s = (ph - self.DUTY) / (1.0 - self.DUTY)            # 0..1 through the swing
                vbody = np.array([v[0], v[1], 0.0])
                tgt = nom + vc * (Ts / 2) + self.K_RAIBERT * (vbody - vc)     # Raibert: half a stance ahead
                tgt[0] += -wz * nom[1] * Ts / 2
                tgt[1] += wz * nom[0] * Ts / 2
                e = 0.5 - 0.5 * math.cos(math.pi * min(1.0, s * 1.15))
                f = self.lift[p] + (tgt - self.lift[p]) * e
                f[2] = nom[2] + self.CLEAR * math.sin(math.pi * s)
                self.foot[p] = f
                q = leg.ik(Rrp.T @ f)
                self.torques(leg, q, self.KP_SW, self.KD_SW)

    # --- the scripted jump ---------------------------------------------------------------------------
    def step_jump(self, dt):
        js, jp, d = self.jump_state, self.jp, self.d
        t = d.time - js["t0"]
        R, Rrp, v, w, roll, pitch = self.body()
        con = self.contacts()
        weight = self.mass * 9.81 / 4
        c = np.array([jp["X_C"], 0.0, -jp["D_CROUCH"]])
        if js["phase"] == "crouch" and t > jp["T_CROUCH"] + 0.15:
            js["phase"], js["t1"] = "push", t
        elif js["phase"] == "push":
            js["off"] = 0 if any(con.values()) else js["off"] + 1
            if js["off"] >= 15 or t - js["t1"] > 0.4:
                js["phase"], js["t1"] = "flight", t
                js["pose_off"] = {p: tuple(d.qpos[a] for a in self.legs[p].ja) for p in M.PODS}
        elif js["phase"] == "flight" and any(con.values()) and t - js["t1"] > 0.04:
            js["phase"], js["t1"] = "land", t
        elif js["phase"] == "land" and t - js["t1"] > jp["LAND_GIVE"] + 1.0:
            self.jump_state = None                                   # standing again: back to walking
            self.foot = {p: np.array([self.legs[p].hip_x, self.legs[p].leg_y, self.legs[p].hip_z - self.h]) for p in M.PODS}
            self.walking = False
            return
        th = math.radians(jp["THETA"])
        for p, leg in self.legs.items():
            hip = np.array([leg.hip_x, leg.leg_y, leg.hip_z])
            rest = hip + np.array([0.0, 0.0, -self.h])
            if js["phase"] == "crouch":
                s = min(t / jp["T_CROUCH"], 1.0)
                s = s * s * (3 - 2 * s)
                f = rest + (hip + c - rest) * s
                self.torques(leg, leg.ik(Rrp.T @ f), (80.0, 80.0, 80.0), (1.5, 1.5, 1.5),
                             Rrp.T @ np.array([0.0, 0.0, weight]))
            elif js["phase"] == "push":
                phi, qh, qk = (d.qpos[a] for a in leg.ja)
                foot, J = leg.fk(phi, qh, qk)
                ext = jp["D_EXT"] + (jp.get("FRONT_EXTRA", 0.0) if leg.hip_x > 0 else 0.0)
                rel = foot - hip
                if con[p] and math.hypot(rel[0], rel[2]) < ext:
                    fdir = Rrp.T @ np.array([-math.cos(th), 0.0, math.sin(th)])     # ground on foot: toward -X, up
                    tj = -(J.T @ fdir)
                    to = np.array([tj[0], tj[1] - tj[2], tj[2]])
                    dout = (d.qvel[leg.jv[0]], d.qvel[leg.jv[1]], d.qvel[leg.jv[1]] + d.qvel[leg.jv[2]])
                    k = 1e9
                    for i in (1, 2):
                        lim = M.TAU * min(1.0, max(0.0, 2.0 * (1.0 - abs(dout[i]) / M.W0)))
                        hi = min(lim, M.A_PUSH) if i == 2 else lim
                        if abs(to[i]) > 1e-9:
                            k = min(k, (hi if to[i] > 0 else lim) / abs(to[i]))
                    bias = (jp.get("SPLIT", 0.0) + jp.get("K_PITCH", 0.0) * (pitch + 0.1 * w[1])) * (1.0 if leg.hip_x > 0 else -1.0)
                    tau = to * k * min(max(1.0 + bias, 0.3), 1.0)            # the front/rear split steers pitch
                    tau[0] = 60.0 * (0.0 - phi) - 1.5 * dout[0]
                    for i in range(3):
                        d.ctrl[leg.act[i]] = tau[i] - (jp.get("KD_PUSH", 0.45) * dout[i] if i else 0.0)
                else:
                    self.torques(leg, (0.0, qh, qk), (60.0, 40.0, 40.0), (1.5, 1.0, 1.0))
            else:
                ang = math.radians(jp["LAND_ANG"]) + pitch            # the landing pose, set in the world frame
                land_rel = np.array([jp["D_LAND"] * math.cos(ang), 0.0, jp["D_LAND"] * math.sin(ang)])
                land = leg.ik(Rrp.T @ (hip + land_rel))
                if js["phase"] == "flight":
                    s = min((t - js["t1"]) / 0.15, 1.0)
                    s = s * s * (3 - 2 * s)
                    q0 = js["pose_off"][p]
                    q = tuple(a + (b - a) * s for a, b in zip(q0, land))
                    self.torques(leg, q, (60.0, 60.0, 60.0), (1.5, 1.5, 1.5))
                else:
                    tl = t - js["t1"]
                    s = 0.0 if tl < jp["LAND_GIVE"] else min((tl - jp["LAND_GIVE"]) / 0.6, 1.0)
                    s = s * s * (3 - 2 * s)
                    q_rest = leg.ik(Rrp.T @ rest)
                    q = tuple(a + (b - a) * s for a, b in zip(land, q_rest))
                    kp = jp["KP_LAND"] + s * (70.0 - jp["KP_LAND"])
                    self.torques(leg, q, (60.0, kp, kp), (1.5, jp["KD_LAND"], jp["KD_LAND"]),
                                 Rrp.T @ np.array([0.0, 0.0, weight]) if con[p] else None)
