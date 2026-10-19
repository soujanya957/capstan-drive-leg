"""A scripted standing long jump, no learning: foot trajectories through the leg's inverse kinematics, tracked
by joint PD with the same motor limits as the RL envs (envs.QuadBase: torque-speed curve, output A's 25 N*m
push limit, the knee's software range).

Phases (each foot's target is a point in the leg's plane, relative to its hip):
  stand    feet under the hips (the rest pose)
  crouch   slowly down and back along the push line, to leg length D_CROUCH
  push     a step to D_EXT along the push line (THETA above horizontal, toward -X): the PD saturates, so this
           is full torque; ends when the feet leave the ground
  flight   swing to the landing pose, set in the WORLD frame (so body pitch does not change where feet land)
  land     at the first touchdown: soft gains, the legs give over LAND_GIVE s, then back to standing

Run: python scripted_jump.py [episodes]      -> distance, landings and part loads, as eval_jump.py
     python scripted_jump.py search [n]      -> random search over the parameters, best written to scripted_jump.json
"""
import json
import math
import os
import sys

import mujoco
import numpy as np

import envs
import quad_model as qm

HERE = os.path.dirname(os.path.abspath(__file__))
L = qm.LEG_L
KNEE_SOFT = math.radians(52.0)    # kept inside the +-60 deg knee range (a software limit on the robot)

P0 = dict(D_CROUCH=0.24, X_C=-0.05, D_EXT=0.385, THETA=62.0, V_PUSH=3.0, T_CROUCH=0.6, KP_PUSH=250.0, KD_PUSH=0.2,
          LAND_ANG=-100.0, D_LAND=0.33, KP_LAND=30.0, KD_LAND=1.6, LAND_GIVE=0.25, FRONT_EXTRA=0.0, K_PITCH=3.0)


def ik(x, z):
    """Foot (x, z) relative to the hip, leg plane, m -> outputs (B, A), rad. Knee on the rest pose's branch."""
    d = min(math.hypot(x, z), 2 * L * 0.999)
    phi = math.atan2(z, x)
    al = math.acos(d / (2 * L))
    th, sh = phi + al, phi - al                                    # thigh, shin angles (from +X toward +Z)
    return math.radians(qm.Q0[0]) - th, math.radians(qm.Q0[1]) - sh


def polar(d, ang_deg):
    return d * math.cos(math.radians(ang_deg)), d * math.sin(math.radians(ang_deg))


def fk(B, A):
    """Outputs (B, A) -> foot (x, z) relative to the hip, and the Jacobian d(x, z)/d(B, A)."""
    th, sh = math.radians(qm.Q0[0]) - B, math.radians(qm.Q0[1]) - A
    p = L * np.array([math.cos(th) + math.cos(sh), math.sin(th) + math.sin(sh)])
    J = L * np.array([[math.sin(th), math.sin(sh)], [-math.cos(th), -math.cos(sh)]])
    return p, J


def knee_ok(B, A):
    """Clamp output A so the knee (A - B) stays inside its soft range."""
    return B, B + min(max(A - B, -KNEE_SOFT), KNEE_SOFT)


class Jumper:
    def __init__(self, env, p):
        self.e, self.p = env, dict(P0, **p)
        self.phase, self.t_phase = "stand", 0.0
        self.t, self.t0 = 0.0, env.data.time
        self.rest = ik(0.0, -2 * L * math.sin(math.radians(60.0)))
        self.weight = float(env.model.body_subtreemass[1]) * 9.81
        self.ext = [self.p["D_EXT"] + (self.p["FRONT_EXTRA"] if pod[1] == "F" else 0.0) for pod in qm.PODS]

    def pitch(self):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.e.data.qpos[3:7])
        R = R.reshape(3, 3)
        return math.atan2(R[0, 2], R[2, 2])                         # rotation about +Y (MuJoCo's sense)

    def targets(self):
        """(B, A) per pod and the gains, for the current phase."""
        p, tp = self.p, self.t - self.t_phase
        c = np.array([p["X_C"], -p["D_CROUCH"]])                     # crouched foot, relative to the hip
        if self.phase == "stand":
            tgt = [self.rest] * 4
            kp, kd = 60.0, 1.5
        elif self.phase == "crouch":
            s = min(tp / p["T_CROUCH"], 1.0)
            s = s * s * (3 - 2 * s)
            a = ik(*c)
            tgt = [tuple((1 - s) * r + s * c_ for r, c_ in zip(self.rest, a))] * 4
            kp, kd = 80.0, 1.5
        elif self.phase == "push":                                  # the foot runs out along the push line at V_PUSH:
            u = np.array([math.cos(math.radians(p["THETA"])), -math.sin(math.radians(p["THETA"]))])
            tgt = []                                                # the body goes the other way, THETA above horizontal
            for pod in qm.PODS:
                ext = p["D_EXT"] + (p["FRONT_EXTRA"] if pod[1] == "F" else 0.0)
                cu = c @ u
                s_max = -cu + math.sqrt(max(cu * cu - c @ c + ext * ext, 0.0))
                tgt.append(ik(*(c + min(p["V_PUSH"] * tp, s_max) * u)))
            kp, kd = p["KP_PUSH"], p["KD_PUSH"]
        else:                                                       # flight and landing: landing pose in the world frame
            ang = p["LAND_ANG"] + math.degrees(self.pitch())        # body pitched nose-down -> feet further forward
            land = ik(*polar(p["D_LAND"], ang))
            if self.phase == "flight":                              # a smooth swing from the take-off pose
                s = min(tp / 0.15, 1.0)
                s = s * s * (3 - 2 * s)
                tgt = [tuple((1 - s) * o + s * l for o, l in zip(o4, land)) for o4 in self.pose_off]
                kp, kd = 60.0, 1.5
            else:
                s = 0.0 if tp < p["LAND_GIVE"] else min((tp - p["LAND_GIVE"]) / 0.6, 1.0)
                s = s * s * (3 - 2 * s)
                tgt = [tuple((1 - s) * l + s * r for l, r in zip(land, self.rest))] * 4
                kp = p["KP_LAND"] + s * (60.0 - p["KP_LAND"])
                kd = p["KD_LAND"]
        return [knee_ok(B, A) for B, A in tgt], kp, kd

    def apply(self, _target):
        """Called by env.sim at every physics step (500 Hz), in place of the RL PD."""
        e = self.e
        self.t = e.data.time - self.t0                              # from the start of this jump
        contact, _ = e.foot_contacts()
        if self.phase == "stand" and self.t >= 0.4:
            self.phase, self.t_phase = "crouch", self.t
        elif self.phase == "crouch" and self.t - self.t_phase >= self.p["T_CROUCH"] + 0.15:
            self.phase, self.t_phase = "push", self.t
        elif self.phase == "push":
            self.off = 0 if contact.any() else getattr(self, "off", 0) + 1
            if self.off >= 15 or self.t - self.t_phase > 0.4:       # all four feet off for 30 ms: take-off
                self.phase, self.t_phase = "flight", self.t
                pos = e.outputs()[0]
                self.pose_off = [(pos[3 * i + 1], pos[3 * i + 2]) for i in range(4)]
        elif self.phase == "flight" and contact.any() and self.t - self.t_phase > 0.04:
            self.phase, self.t_phase = "land", self.t
        tgt, kp, kd = self.targets()
        target = np.zeros(12)
        for i, (B, A) in enumerate(tgt):
            target[3 * i + 1], target[3 * i + 2] = B, A
        pos, vel = e.outputs()
        tau = kp * (target - pos) - kd * vel
        tau[0::3] = 60.0 * (0.0 - pos[0::3]) - 1.5 * vel[0::3]       # ab/ad held at 0
        lim = qm.TAU * np.clip(2.0 * (1.0 - np.abs(vel) / qm.W0), 0.0, 1.0)
        w = self.weight / 4.0
        p = self.p
        for i, pod in enumerate(qm.PODS):
            sl = slice(3 * i + 1, 3 * i + 3)
            foot, J = fk(*pos[sl])
            if self.phase in ("stand", "crouch") or (self.phase == "land" and contact[i]):
                tau[sl] += -J.T @ np.array([0.0, w])                 # carry the leg's share of the weight
            elif self.phase == "push" and contact[i] and np.linalg.norm(foot) < self.ext[i]:   # a foot already off is not pushed
                # force control: the ground's push on the foot along THETA (toward -X, up), as large as the
                # motors allow; the front/rear split steers the body's pitch
                th = math.radians(p["THETA"])
                f = np.array([-math.cos(th), math.sin(th)])
                pitch, wy = self.pitch(), e.data.qvel[4]
                bias = p["K_PITCH"] * (pitch + 0.1 * wy) * (1.0 if pod[1] == "F" else -1.0)
                t1 = -J.T @ f
                hi = np.array([lim[3 * i + 1], min(lim[3 * i + 2], envs.A_PUSH)])
                lo = -lim[sl]
                k = min([(hi[j] if t1[j] > 0 else lo[j]) / t1[j] for j in range(2) if abs(t1[j]) > 1e-9])
                tau[sl] = t1 * k * float(np.clip(1.0 + bias, 0.3, 1.0)) - p["KD_PUSH"] * vel[sl]
        lim = qm.TAU * np.clip(2.0 * (1.0 - np.abs(vel) / qm.W0), 0.0, 1.0)
        hi = lim.copy()
        hi[2::3] = np.minimum(hi[2::3], envs.A_PUSH)
        tau = np.clip(tau, -lim, hi)
        e.data.ctrl[e.act.ravel()] = tau
        return tau


def run(p, seed, masses, track=False, T=3.5, record=None):
    e = envs.JumpEnv(seed=seed, masses=masses)
    e.reset(seed=seed)
    e.data.time = 0.0
    j = Jumper(e, p)
    e.apply = j.apply
    if track:
        e.track, e.peak_grf, e.peak_land, e.peak_tau = True, np.zeros(4), np.zeros(4), np.zeros(12)
        e.contact_log = []
    x_off = x_land = None
    bottomed, fell, lim = False, False, 0.0
    peak_grf = 0.0
    n = int(T / (envs.DT * envs.DECIM))
    for k in range(n):
        ph0 = j.phase
        e.took_off = j.phase in ("flight", "land")
        e.grf_on = True
        e.sim(np.zeros(12))
        if ph0 == "push" and j.phase == "flight":
            x_off = e.data.qpos[0]
        if ph0 == "flight" and j.phase == "land":
            x_land = e.data.qpos[0]
        _, bad = e.foot_contacts()
        bottomed |= bad
        lim = max(lim, e.step_lim)
        if j.phase == "land":
            peak_grf = max(peak_grf, e.step_grf)
        R, v, w, grav = e.base()
        if -grav[2] < 0.5:
            fell = True
            break
        if record is not None:
            record(e, j)
    R, v, w, grav = e.base()
    dist = -(x_land - x_off) if (x_off is not None and x_land is not None) else 0.0
    stood = not fell and not bottomed and j.phase == "land" and -grav[2] > 0.95 and \
        abs(e.data.qpos[2] - qm.STAND_Z) < 0.05 and np.linalg.norm(e.data.qvel[:3]) < 0.1
    travel = -(e.data.qpos[0])
    return dict(distance=dist, travel=travel, stood=bool(stood), bottomed=bool(bottomed), fell=bool(fell),
                land_grf=peak_grf, lim=lim, env=e)


def score(p, masses, seeds):
    rs = [run(p, s, masses) for s in seeds]
    ok = sum(r["stood"] for r in rs)
    d = np.mean([r["distance"] for r in rs])
    g = max(r["land_grf"] for r in rs)
    lim = max(r["lim"] for r in rs)
    return d * (ok / len(rs)) ** 2 - 1e-4 * max(0.0, g - 500.0) - 0.004 * lim, ok, d, g, lim


SPACE = dict(D_CROUCH=(0.22, 0.28), X_C=(-0.15, 0.05), V_PUSH=(1.5, 5.0), D_EXT=(0.36, 0.395), KP_PUSH=(120.0, 400.0),
             KD_PUSH=(0.0, 1.0), LAND_ANG=(-120.0, -80.0), D_LAND=(0.29, 0.37), KP_LAND=(15.0, 60.0),
             KD_LAND=(0.8, 3.0), K_PITCH=(0.0, 8.0), THETA=(55.0, 75.0), LAND_GIVE=(0.1, 0.4), FRONT_EXTRA=(-0.03, 0.01))


def _score(args):
    return score(*args)


def search(n, masses):
    """Random local search, 12 candidates at a time in parallel, each scored on 8 randomised seeds."""
    from multiprocessing import Pool
    rng = np.random.default_rng(0)
    seeds = range(8)
    best_p = params()
    pool = Pool(12)
    best = score(best_p, masses, seeds)
    print("start", [round(float(x), 3) for x in best], flush=True)
    for i in range(n):
        cands = []
        for _ in range(12):
            if rng.random() < 0.15:
                c = {k: rng.uniform(*v) for k, v in SPACE.items()}
            else:                                                   # a step around the best
                sc = rng.uniform(0.03, 0.2)
                c = {k: float(np.clip(best_p[k] + rng.normal(0, sc) * (v[1] - v[0]), *v)) for k, v in SPACE.items()}
            cands.append(dict(P0, **c))
        for c, s in zip(cands, pool.map(_score, [(c, masses, seeds) for c in cands])):
            if s[0] > best[0]:
                best, best_p = s, c
                print(i, "best", [round(float(x), 3) for x in s], {k: round(v, 3) for k, v in c.items()}, flush=True)
                json.dump(best_p, open(os.path.join(HERE, "scripted_jump.json"), "w"), indent=1)
    return best_p


def params():
    f = os.path.join(HERE, "scripted_jump.json")
    return dict(P0, **json.load(open(f))) if os.path.exists(f) else dict(P0)


def evaluate(n, masses):
    from eval_jump import leg_loads
    p = params()
    rows = []
    for s in range(n):
        r = run(p, 500 + s, masses, track=True)
        r["loads"] = leg_loads(r["env"])
        r["grf"], r["land"], r["tau"] = r["env"].peak_grf.max(), r["env"].peak_land.max(), r["env"].peak_tau.max()
        rows.append(r)
    d = np.array([r["distance"] for r in rows])
    tr = np.array([r["travel"] for r in rows])
    ok = sum(r["stood"] for r in rows)
    weight = masses["total"] * 9.81
    print(f"{n} scripted jumps: in the air {d.mean():.2f} m (min {d.min():.2f}, max {d.max():.2f}); total travel "
          f"{tr.mean():.2f} m; landed and stood (body never touching) {ok}/{n}")
    grf = max(r["grf"] for r in rows)
    land = max(r["land"] for r in rows)
    print(f"peak foot force: push-off {grf:.0f} N, landing {land:.0f} N ({land / weight:.1f} x body weight on one foot)")
    print(f"peak output torque {max(r['tau'] for r in rows):.1f} N*m of {qm.TAU}; worst joint-limit load "
          f"{max(r['lim'] for r in rows):.1f} N*m")
    ld = {k: max(r["loads"][k] for r in rows) for k in rows[0]["loads"]}
    print(f"  coupler / pivots {ld['coupler']:.0f} N (636ZZ C0 1370: s0 {1370 / max(ld['coupler'], 1):.2f}); pushing "
          f"{max(ld['coupler_push'], 0):.0f} N (buckling ~1020: SF {1020 / max(ld['coupler_push'], 1):.1f})")
    print(f"  knee pin {ld['knee']:.0f} N (6000ZZ C0 1960: s0 {1960 / max(ld['knee'], 1):.2f})")
    print(f"  hub insert pull-out {ld['hub_propped']:.0f} N with the trunnion, {ld['hub_bound']:.0f} N without")
    json.dump(dict(params=p, distance=d.tolist(), travel=tr.tolist(), stood=[r["stood"] for r in rows],
                   peak_grf=grf, peak_land=land, loads=ld),
              open(os.path.join(HERE, "policies", "scripted_jump_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    masses = json.load(open(os.path.join(HERE, "masses.json")))
    if len(sys.argv) > 1 and sys.argv[1] == "search":
        search(int(sys.argv[2]) if len(sys.argv) > 2 else 200, masses)
    else:
        evaluate(int(sys.argv[1]) if len(sys.argv) > 1 else 20, masses)
