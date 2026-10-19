"""Evaluate the trained jump policy: distance, flight time, landing, and the loads it puts on the hardware
(peak contact force per foot, peak output torque against the 31.5 N*m limit), over several seeds with the
training's domain randomisation. These landing loads are the ones the strength checks are compared with.

Run: python eval_jump.py [episodes]
"""
import json
import os
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import envs
import quad_model as qm

HERE = os.path.dirname(os.path.abspath(__file__))


sys.path.insert(0, os.path.join(HERE, "..", "..", "v9_1_buildable", "src"))
import generate as g                                                   # noqa: E402


def leg_loads(e):
    """Peak loads in v9.1's parts over an episode's foot contacts: the 4-bar statics of v9.1's check_strength /
    fea_check (coupler = crank and lever pivots, knee pin) from the foot force in the leg's plane and the
    leg's pose; the hub screws' insert pull-out with the pod propped on its trunnion (fixed at the hub's
    bearings, pinned at the trunnion) and, as an upper bound, with no trunnion (check_strength.py's case)."""
    u = lambda a: np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
    cross = lambda a, b: a[0] * b[1] - a[1] * b[0]
    x_a, x_b = sum(g.AB_X["inner"]) / 2, sum(g.HF_X) / 2               # hub bearings, trunnion (pod frame, mm)
    L = x_b - x_a
    per_bolt = g.AB_HUB_R / (g.AB_HUB_N * g.AB_HUB_R ** 2 / 2)          # insert pull per N*mm of hub moment
    m = e.model
    out = dict(coupler=0.0, knee=0.0, hub_propped=0.0, hub_bound=0.0, coupler_push=0.0)
    for j, fw, qpos, phase in e.contact_log:
        p = qm.PODS[j]
        R = np.zeros(9)
        import mujoco
        mujoco.mju_quat2Mat(R, qpos[3:7])
        fb = R.reshape(3, 3).T @ fw                                     # body frame
        hip, knee = qpos[m.joint(f"{p}_hip").qposadr[0]], qpos[m.joint(f"{p}_knee").qposadr[0]]
        qt, qs = qm.Q0[0] - np.degrees(hip), qm.Q0[1] - np.degrees(hip + knee)
        k, f, c, d = (np.array(pt, float) for pt in g.leg4_points(qt, qs))
        F = np.array([fb[0], fb[2]])                                    # in the leg's plane (x, z)
        uc = (d - c) / np.linalg.norm(d - c)
        fc = -cross(f - k, F) / cross(d - k, uc)                        # + pushes the shin: coupler in compression
        fk = -(F + fc * uc)
        out["coupler"] = max(out["coupler"], abs(fc))
        out["coupler_push"] = max(out["coupler_push"], fc)
        out["knee"] = max(out["knee"], float(np.linalg.norm(fk)))
        P = float(np.hypot(fb[1], fb[2]))                               # across the pod's axis
        a_ = min(max(f[0] - x_a, 0.0), L)
        b_ = L - a_
        M_prop = P * b_ * (L * L - b_ * b_) / (2 * L * L)
        out["hub_propped"] = max(out["hub_propped"], M_prop * per_bolt)
        out["hub_bound"] = max(out["hub_bound"], P * (f[0] - x_a) * per_bolt)
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    masses = json.load(open(os.path.join(HERE, "masses.json")))
    pol = os.environ.get("EVAL_POLICY", os.path.join(HERE, "policies", "jump"))          # or a checkpoint
    vn = os.environ.get("EVAL_VECNORM", os.path.join(HERE, "policies", "jump_vecnorm.pkl"))
    model = PPO.load(pol, device="cpu")
    rows = []
    for seed in range(n):
        v = VecNormalize.load(vn,
                              DummyVecEnv([lambda s=seed: envs.JumpEnv(seed=500 + s, masses=masses)]))
        v.training, v.norm_reward = False, False
        e = v.envs[0]
        o = v.reset()
        e.track, e.peak_grf, e.peak_land, e.peak_tau = True, np.zeros(4), np.zeros(4), np.zeros(12)
        e.contact_log = []
        info, fell = {}, False
        while True:
            a, _ = model.predict(o, deterministic=True)
            o, r, d, inf = v.step(a)
            if d[0]:
                info = inf[0]
                fell = not info.get("TimeLimit.truncated", False)          # ended early: it fell
                break
        rows.append(dict(distance=info.get("distance", 0.0), flight=info.get("flight", 0.0), landed=bool(info.get("landed")),
                         bottomed=bool(info.get("bottomed")),
                         fell=fell, grf=e.peak_grf.copy(), land=e.peak_land.copy(), tau=e.peak_tau.copy(),
                         loads=leg_loads(e)))
    weight = masses["total"] * 9.81
    d = np.array([r["distance"] for r in rows])
    fl = np.array([r["flight"] for r in rows])
    ok = np.array([r["landed"] and not r["fell"] and not r["bottomed"] for r in rows])     # on its feet, body never down
    grf = np.array([r["grf"].max() for r in rows])
    land = np.array([r["land"].max() for r in rows])
    tau = np.array([r["tau"].max() for r in rows])
    print(f"{n} jumps: distance {d.mean():.2f} m (min {d.min():.2f}, max {d.max():.2f}), flight {fl.mean():.2f} s, "
          f"landed on its feet and stood (body never touching) {ok.sum()}/{n}")
    print(f"peak foot force, push-off: mean {grf.mean():.0f} N, max {grf.max():.0f} N ({grf.max() / weight:.1f} x body weight on one foot)")
    print(f"peak foot force, landing:  mean {land.mean():.0f} N, max {land.max():.0f} N ({land.max() / weight:.1f} x body weight on one foot)")
    print(f"peak output torque: max {tau.max():.1f} N*m of {qm.TAU:.1f}")
    ld = {k: max(r["loads"][k] for r in rows) for k in rows[0]["loads"]}
    print(f"v9.1 parts, worst over the {n} jumps (from the 4-bar statics):")
    print(f"  coupler / crank and lever pivots {ld['coupler']:.0f} N (636ZZ C0 1370: s0 {1370 / max(ld['coupler'], 1):.2f}); "
          f"pushing {max(ld['coupler_push'], 0):.0f} N (buckling ~1020: SF {1020 / max(ld['coupler_push'], 1):.1f})")
    print(f"  knee pin {ld['knee']:.0f} N (6000ZZ C0 1960: s0 {1960 / max(ld['knee'], 1):.2f})")
    print(f"  hub insert pull-out {ld['hub_propped']:.0f} N with the trunnion, {ld['hub_bound']:.0f} N without "
          f"(1000 N each: SF {1000 / max(ld['hub_propped'], 1):.1f} / {1000 / max(ld['hub_bound'], 1):.1f})")
    json.dump(dict(distance=d.tolist(), flight=fl.tolist(), landed=ok.tolist(), peak_grf=grf.tolist(), peak_land=land.tolist(), peak_tau=tau.tolist(), loads=[r["loads"] for r in rows]),
              open(os.path.join(HERE, "policies", "jump_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
